"""
Transaction graph builder.
Converts tabular transaction data into a PyTorch Geometric graph
for GraphSAGE/GAT training.

Graph design:
  - Nodes = accounts (sender + receiver = same node space)
  - Edges = transactions (directed: sender → receiver)
  - Node features = aggregated transaction statistics per account
  - Edge features = transaction amount, currency flags, format
  - Edge label = Is Laundering (for edge-level classification)

Optimized for millions of transactions using vectorized pandas operations.
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional

logger = logging.getLogger(__name__)

# PyTorch Geometric imports (guarded for environments without GPU setup)
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/PyG not available — graph features will use NetworkX only.")


def build_transaction_graph_nx(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from transaction DataFrame.
    Optimized using nx.from_pandas_edgelist for performance on large datasets.
    """
    logger.info("Building NetworkX transaction graph (vectorized)...")
    
    # Map raw columns to engineered equivalents if needed
    source_col = "Account"
    target_col = "Account.1"
    
    edge_attr = []
    for col in ["amount_paid", "label", "Payment Format", "is_cross_currency"]:
        if col in df.columns:
            edge_attr.append(col)

    G = nx.from_pandas_edgelist(
        df,
        source=source_col,
        target=target_col,
        edge_attr=edge_attr,
        create_using=nx.DiGraph()
    )

    logger.info(
        f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
    )
    
    # Fast count of fraud edges
    fraud_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("label", d.get("is_fraud", 0)))
    logger.info(f"Fraud edges: {fraud_edges:,} ({fraud_edges/G.number_of_edges()*100:.2f}%)")
    return G


def build_pyg_data(df: pd.DataFrame, test_size: float = 0.2) -> Optional["Data"]:
    """
    Build a PyTorch Geometric Data object for GNN training.
    Fully vectorized to support datasets with millions of rows in seconds.

    Returns:
        PyG Data object, or None if PyG is not installed.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch Geometric not available. Cannot build PyG Data.")
        return None

    logger.info("Building PyTorch Geometric graph (vectorized)...")

    # ── Build account → node index mapping ───────────────────────────────────
    all_accounts = pd.concat([
        df["Account"].astype(str),
        df["Account.1"].astype(str),
    ]).unique()
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    n_nodes = len(account_to_idx)
    logger.info(f"Unique accounts (nodes): {n_nodes:,}")

    # ── Node features: per-account aggregated statistics ──────────────────────
    # Vectorized Sent and Received Aggregates
    df_amt_paid = pd.to_numeric(df.get("amount_paid", df.get("Amount Paid", 0)), errors="coerce").fillna(0)
    df_amt_recv = pd.to_numeric(df.get("amount_received", df.get("Amount Received", 0)), errors="coerce").fillna(0)
    
    # Aggregate data using vector Pandas groupby
    sent_df = pd.DataFrame({"Account": df["Account"], "amount_paid": df_amt_paid})
    recv_df = pd.DataFrame({"Account.1": df["Account.1"], "amount_received": df_amt_recv})
    
    sent_agg = sent_df.groupby("Account")["amount_paid"].agg(["sum", "count", "mean", "std"]).fillna(0)
    recv_agg = recv_df.groupby("Account.1")["amount_received"].agg(["sum", "count"]).fillna(0)

    # Vectorized sent_to and recv_from unique neighbor counts (degree)
    # Replaces the old O(N^2) loop bottleneck!
    sent_to_counts = df.groupby("Account")["Account.1"].nunique()
    recv_from_counts = df.groupby("Account.1")["Account"].nunique()

    # Join everything onto the unique accounts list
    nodes_df = pd.DataFrame(index=all_accounts)
    nodes_df = nodes_df.join(sent_agg.rename(columns={
        "sum": "sent_sum", "count": "sent_count", "mean": "sent_mean", "std": "sent_std"
    }))
    nodes_df = nodes_df.join(recv_agg.rename(columns={
        "sum": "recv_sum", "count": "recv_count"
    }))
    nodes_df = nodes_df.join(sent_to_counts.rename("sent_to_unique"))
    nodes_df = nodes_df.join(recv_from_counts.rename("recv_from_unique"))
    
    nodes_df = nodes_df.fillna(0)

    # Convert to feature matrix
    node_features = np.zeros((n_nodes, 8), dtype=np.float32)
    node_features[:, 0] = np.log1p(nodes_df["sent_sum"].values)
    node_features[:, 1] = nodes_df["sent_count"].values
    node_features[:, 2] = np.log1p(nodes_df["sent_mean"].values)
    node_features[:, 3] = np.log1p(nodes_df["sent_std"].values)
    node_features[:, 4] = np.log1p(nodes_df["recv_sum"].values)
    node_features[:, 5] = nodes_df["recv_count"].values
    node_features[:, 6] = nodes_df["sent_to_unique"].values
    node_features[:, 7] = nodes_df["recv_from_unique"].values

    # Normalize node features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)

    # ── Edge index and features ───────────────────────────────────────────────
    src_nodes = df["Account"].astype(str).map(account_to_idx).values
    dst_nodes = df["Account.1"].astype(str).map(account_to_idx).values

    edge_index = torch.tensor(
        np.stack([src_nodes, dst_nodes], axis=0),
        dtype=torch.long,
    )

    # Edge features: log_amount, is_cross_currency, payment_format_enc
    edge_feats = np.zeros((len(df), 4), dtype=np.float32)
    edge_feats[:, 0] = np.log1p(df_amt_paid.values)
    edge_feats[:, 1] = df.get("is_cross_currency", pd.Series(np.zeros(len(df)))).values.astype(float)
    edge_feats[:, 2] = df.get("payment_format_enc", pd.Series(np.zeros(len(df)))).values.astype(float)
    edge_feats[:, 3] = df.get("is_cross_bank", pd.Series(np.zeros(len(df)))).values.astype(float)

    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    # Edge labels
    labels = df.get("label", df.get("Is Laundering", pd.Series(np.zeros(len(df))))).values
    edge_labels = torch.tensor(labels.astype(int), dtype=torch.long)

    # ── Train/test split masks ────────────────────────────────────────────────
    n_edges = len(df)
    indices = np.random.permutation(n_edges)
    n_test = int(n_edges * test_size)
    test_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_mask[indices[:n_test]] = True
    train_mask = ~test_mask

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_labels,
        train_mask=train_mask,
        test_mask=test_mask,
        num_nodes=n_nodes,
    )

    # Store the account mapping for subgraph extraction
    data.account_to_idx = account_to_idx
    data.idx_to_account = {v: k for k, v in account_to_idx.items()}

    logger.info(
        f"PyG Data: {data.num_nodes:,} nodes, {data.num_edges:,} edges, "
        f"node_feat_dim={data.x.shape[1]}, edge_feat_dim={data.edge_attr.shape[1]}"
    )
    return data


def get_fraud_subgraph(
    G: nx.DiGraph,
    account_id: str,
    hops: int = 2,
) -> nx.DiGraph:
    """
    Extract the local subgraph around an account (for explainability).
    Returns a subgraph showing all accounts within `hops` transactions.
    """
    nodes_in_subgraph = {account_id}
    frontier = {account_id}

    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            if node in G:
                next_frontier.update(G.predecessors(node))
                next_frontier.update(G.successors(node))
        frontier = next_frontier - nodes_in_subgraph
        nodes_in_subgraph.update(frontier)

    return G.subgraph(nodes_in_subgraph).copy()
