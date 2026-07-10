"""
XGBoost / LightGBM baseline classifier for fraud detection.

IMPORTANT: This dataset is severely imbalanced (~1% fraud).
- Never report plain accuracy — it's meaningless
- Use: precision, recall, PR-AUC, F1, and recall@fixed-FPR
- Use class weighting and/or SMOTE for handling imbalance
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

logger = logging.getLogger(__name__)

# Saved model path
MODEL_PATH = Path("models/saved/baseline_xgb.pkl")


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "FraudGraph-Baseline",
    tracking_uri: str = "./mlflow_runs",
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train an XGBoost baseline with proper imbalance handling.
    Logs all params, metrics, and the model artifact to MLflow.

    Returns:
        (trained_model, metrics_dict)
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Class weight for imbalance
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"Training XGBoost | class ratio: {scale_pos_weight:.1f}:1 (neg:pos)")

    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",  # fast, CPU-friendly
    }

    with mlflow.start_run(run_name="xgb_baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate_train", float(y_train.mean()))

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluation ─────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_classifier(y_test, y_prob, y_pred)
        mlflow.log_metrics(metrics)

        # Save model artifact
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        mlflow.xgboost.log_model(model, artifact_path="xgb_baseline")

        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

    _log_metrics_table(metrics, model_name="XGBoost Baseline")
    return model, metrics


def evaluate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    fixed_fpr: float = 0.05,
) -> dict:
    """
    Compute the full suite of fraud detection metrics.
    Uses PR-AUC as the primary metric (not ROC-AUC, not accuracy).

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for fraud class
        y_pred: Binary predictions at 0.5 threshold
        fixed_fpr: FPR budget for recall@fixed-FPR metric

    Returns:
        Dict of metric names → values
    """
    # Core metrics
    pr_auc = average_precision_score(y_true, y_prob)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    # Precision/recall at 0.5 threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Recall at fixed FPR budget (e.g., catch max fraud while keeping FP < 5%)
    recall_at_fixed_fpr = _recall_at_fixed_fpr(y_true, y_prob, fixed_fpr)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision_at_05": round(precision, 4),
        "recall_at_05": round(recall, 4),
        "f1_at_05": round(f1, 4),
        f"recall_at_fpr_{int(fixed_fpr*100)}pct": round(recall_at_fixed_fpr, 4),
        "fraud_rate_test": round(float(y_true.mean()), 6),
        "n_test": len(y_true),
        "n_fraud_test": int(y_true.sum()),
    }
    return metrics


def _recall_at_fixed_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    """
    What recall can we achieve while keeping FPR <= target_fpr?
    This is the metric that matters for fraud operations:
    "We can only review X% of transactions — what % of fraud do we catch?"
    """
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Find the best TPR where FPR <= target_fpr
        valid = fpr <= target_fpr
        if not valid.any():
            return 0.0
        return float(tpr[valid].max())
    except Exception:
        return 0.0


def _log_metrics_table(metrics: dict, model_name: str) -> None:
    """Print a clean metrics table."""
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  PR-AUC (primary):      {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:               {metrics['roc_auc']:.4f}")
    print(f"  Precision @0.5:        {metrics['precision_at_05']:.4f}")
    print(f"  Recall @0.5:           {metrics['recall_at_05']:.4f}")
    print(f"  F1 @0.5:               {metrics['f1_at_05']:.4f}")
    recall_key = [k for k in metrics if "recall_at_fpr" in k]
    if recall_key:
        k = recall_key[0]
        print(f"  {k}:    {metrics[k]:.4f}")
    print(f"  Test fraud rate:       {metrics['fraud_rate_test']:.4f}")
    print(f"{'='*50}\n")
    print("  NOTE: Plain accuracy is NOT reported because the dataset is")
    print("  severely imbalanced. A model predicting 'all legit' would get")
    print(f"  ~{(1 - metrics['fraud_rate_test'])*100:.1f}% accuracy — meaningless for fraud detection.")
    print(f"{'='*50}\n")


def load_model(model_path: str = str(MODEL_PATH)) -> Optional[xgb.XGBClassifier]:
    """Load a saved baseline model."""
    path = Path(model_path)
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {path}")
    return model
