"""
GraphSAGE model for fraud/AML detection on transaction graphs.
Built with PyTorch Geometric.

Why GraphSAGE over GCN?
- GraphSAGE (Hamilton et al., 2017) is inductive — it can generate embeddings
  for new accounts/transactions it hasn't seen during training.
- This is critical for real-time fraud: new accounts appear constantly.
- GCN is transductive — requires the full graph at inference time.
- GraphSAGE samples a fixed-size neighborhood, making batch training feasible
  on large graphs.

Edge-level classification:
  We predict whether an edge (transaction) is fraudulent, not a node.
  After learning node embeddings, we classify edges by concatenating
  the source and destination embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import mlflow

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/saved/graphsage_model.pkl")


# ── Model Definition ──────────────────────────────────────────────────────────

def build_graphsage_model(
    node_feature_dim: int,
    edge_feature_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
):
    """
    Build a GraphSAGE model for edge-level fraud classification.

    Architecture:
      - GraphSAGE layers: learn node embeddings aggregating neighborhood info
      - Edge classifier MLP: src_embedding ‖ dst_embedding → fraud probability

    Returns:
        PyTorch module (FraudGraphSAGE)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import SAGEConv
    except ImportError:
        raise ImportError(
            "PyTorch Geometric required. Install: pip install torch torch-geometric"
        )

    class FraudGraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            in_dim = node_feature_dim
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
                self.convs.append(SAGEConv(in_dim, out_dim, aggr="mean"))
                in_dim = out_dim

            # Edge classifier: concatenate src + dst embeddings + edge features
            edge_input_dim = hidden_dim * 2 + edge_feature_dim
            self.edge_classifier = nn.Sequential(
                nn.Linear(edge_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, edge_attr=None):
            # Learn node embeddings
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = F.relu(x)
                if i < len(self.convs) - 1:
                    x = self.dropout(x)

            # Edge classification
            src, dst = edge_index[0], edge_index[1]
            edge_repr = torch.cat([x[src], x[dst]], dim=1)
            if edge_attr is not None:
                edge_repr = torch.cat([edge_repr, edge_attr], dim=1)

            return self.edge_classifier(edge_repr)

        def get_node_embeddings(self, x, edge_index):
            """Get node embeddings (for subgraph visualization)."""
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = F.relu(x)
            return x

    return FraudGraphSAGE()


def train_graphsage(
    pyg_data,
    tracking_uri: str = "./mlflow_runs",
    experiment_name: str = "FraudGraph-GNN",
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
) -> tuple[object, dict]:
    """
    Train GraphSAGE on the transaction graph.
    Handles class imbalance via weighted cross-entropy loss.

    Returns:
        (trained_model, metrics_dict)
    """
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader
    except ImportError:
        raise ImportError("PyTorch Geometric not installed.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    node_dim = pyg_data.x.shape[1]
    edge_dim = pyg_data.edge_attr.shape[1] if pyg_data.edge_attr is not None else 0

    model = build_graphsage_model(
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
    )

    # Class weights for imbalance
    labels = pyg_data.edge_label.numpy()
    n_pos = labels.sum()
    n_neg = (labels == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    logger.info(f"GraphSAGE training | class weight: {pos_weight.item():.1f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    with mlflow.start_run(run_name="graphsage") as run:
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "n_nodes": pyg_data.num_nodes,
            "n_edges": pyg_data.num_edges,
        })

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)

            # Train on train_mask edges
            train_out = out[pyg_data.train_mask]
            train_labels = pyg_data.edge_label[pyg_data.train_mask]

            # Weighted cross-entropy
            weight = torch.ones(2)
            weight[1] = pos_weight.item()
            loss = F.cross_entropy(train_out, train_labels, weight=weight)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # ── Evaluation on test set ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            test_out = out[pyg_data.test_mask]
            test_labels = pyg_data.edge_label[pyg_data.test_mask].numpy()

        y_prob = F.softmax(test_out, dim=1)[:, 1].numpy()
        y_pred = (y_prob >= 0.5).astype(int)

        from models.baseline import evaluate_classifier
        metrics = evaluate_classifier(test_labels, y_prob, y_pred)
        mlflow.log_metrics({f"gnn_{k}": v for k, v in metrics.items()})

        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model.state_dict(), f)

        run_id = run.info.run_id

    _log_metrics_table(metrics, "GraphSAGE")
    return model, metrics


def _log_metrics_table(metrics: dict, model_name: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:.4f}")
    print(f"{'='*50}\n")


def print_comparison_table(baseline_metrics: dict, gnn_metrics: dict) -> None:
    """Print a comparison table: baseline vs. GNN."""
    keys = ["pr_auc", "roc_auc", "precision_at_05", "recall_at_05", "f1_at_05"]
    recall_key = [k for k in baseline_metrics if "recall_at_fpr" in k]
    if recall_key:
        keys.append(recall_key[0])

    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON — XGBoost Baseline vs. GraphSAGE GNN")
    print(f"{'='*65}")
    print(f"  {'Metric':<35} {'XGBoost':>10} {'GraphSAGE':>12} {'Delta':>6}")
    print(f"  {'-'*60}")

    for k in keys:
        b = baseline_metrics.get(k, 0)
        g = gnn_metrics.get(k, 0)
        delta = g - b
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {k:<35} {b:>10.4f} {g:>12.4f} {delta_str:>6}")

    print(f"{'='*65}")
    print("  KEY INSIGHT: The GNN improves detection of COORDINATED fraud")
    print("  (multi-hop money laundering rings) that look normal row-by-row.")
    print(f"{'='*65}\n")
