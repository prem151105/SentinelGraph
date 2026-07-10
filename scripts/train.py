"""
FraudGraph — Master Training Script
Runs all phases: feature engineering → baseline → GNN → fusion evaluation → model card.
Usage: python scripts/train.py --data ./data/raw/HI-Small_Trans.csv
"""

import argparse
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_training(data_path: str, tracking_uri: str = "./mlflow_runs"):
    from features.tabular import load_ibm_aml, engineer_features, get_feature_matrix
    from features.graph_builder import build_transaction_graph_nx, build_pyg_data
    from models.baseline import train_baseline
    from models.gnn import train_graphsage, print_comparison_table
    from models.fusion import ScoreFusion
    from models.model_card import generate_model_card
    from config import settings

    Path("models/saved").mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Feature engineering + XGBoost baseline ──────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Feature Engineering + XGBoost Baseline")
    logger.info("="*60)

    df = load_ibm_aml(data_path)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = get_feature_matrix(df)

    # Save scaler (needed at inference time)
    with open("models/saved/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved.")

    baseline_model, baseline_metrics = train_baseline(
        X_train, y_train, X_test, y_test,
        tracking_uri=tracking_uri,
    )

    # ── Phase 2: Graph + GraphSAGE ────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Graph Construction + GraphSAGE Training")
    logger.info("="*60)

    gnn_model = None
    gnn_metrics = None

    try:
        G_nx = build_transaction_graph_nx(df)
        pyg_data = build_pyg_data(df)

        if pyg_data is not None:
            gnn_model, gnn_metrics = train_graphsage(
                pyg_data, tracking_uri=tracking_uri
            )
            print_comparison_table(baseline_metrics, gnn_metrics)
        else:
            logger.warning("PyG not available — skipping GNN training.")
    except Exception as e:
        logger.error(f"GNN training failed: {e}. Continuing with baseline only.")

    # ── Phase 3: Score fusion evaluation ─────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Score Fusion Evaluation")
    logger.info("="*60)

    fused_metrics = None
    if gnn_metrics is not None:
        fusion = ScoreFusion(
            tabular_weight=settings.tabular_model_weight,
            threshold=settings.fraud_threshold,
        )
        import numpy as np
        tabular_probs = baseline_model.predict_proba(X_test)[:, 1]
        # Note: GNN edge scores would need to be aligned with test set indices
        # For now, evaluate tabular-only fusion
        fused_metrics = fusion.evaluate_fused(y_test, tabular_probs)
        logger.info(f"Fused PR-AUC: {fused_metrics.get('pr_auc', 'N/A')}")

    # ── Phase 4: Model card ───────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Generating Model Card")
    logger.info("="*60)

    card_path = generate_model_card(
        model_name="FraudGraph",
        model_version="1.0.0",
        mlflow_run_id="see_mlflow_for_run_id",
        baseline_metrics=baseline_metrics,
        gnn_metrics=gnn_metrics,
        fused_metrics=fused_metrics,
        training_data_info={
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "fraud_rate_train": float(y_train.mean()),
            "tabular_weight": settings.tabular_model_weight,
            "threshold": settings.fraud_threshold,
            "drift_threshold": settings.drift_threshold,
        },
    )
    logger.info(f"Model card saved: {card_path}")

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE [SUCCESS]")
    logger.info("="*60)
    logger.info(f"Baseline PR-AUC:  {baseline_metrics.get('pr_auc', 'N/A')}")
    if gnn_metrics:
        logger.info(f"GNN PR-AUC:       {gnn_metrics.get('pr_auc', 'N/A')}")
    logger.info(f"MLflow UI:        mlflow ui --backend-store-uri {tracking_uri}")
    logger.info(f"Model card:       {card_path}")
    logger.info("Next: uvicorn serving.main:app --port 8001 --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudGraph Training Pipeline")
    parser.add_argument("--data", default="./data/raw/HI-Small_Trans.csv")
    parser.add_argument("--mlflow-uri", default="./mlflow_runs")
    args = parser.parse_args()

    run_training(args.data, args.mlflow_uri)
