"""
Evidently AI drift monitoring + automated retraining trigger.
Compares a rolling window of live transaction features against training distribution.
If drift score exceeds threshold, triggers a new training run and logs to MLflow.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import mlflow

logger = logging.getLogger(__name__)

DRIFT_REPORTS_DIR = Path("mlops/drift_reports")


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Optional[str] = None,
) -> dict:
    """
    Compute Evidently AI drift report comparing reference vs. current data.

    Args:
        reference_df: Training data (reference distribution)
        current_df: Rolling window of recent live traffic
        feature_columns: Feature columns to monitor
        output_path: If given, save HTML report to this path

    Returns:
        Dict with: drift_detected (bool), drift_score (float), drifted_features (list)
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping
    except ImportError:
        logger.warning("Evidently not installed. Falling back to manual PSI calculation.")
        return _manual_psi_drift(reference_df, current_df, feature_columns)

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Only use available feature columns
    available = [c for c in feature_columns if c in reference_df.columns and c in current_df.columns]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[available].copy(),
        current_data=current_df[available].copy(),
    )

    result = report.as_dict()

    # Parse Evidently result
    drift_metrics = result.get("metrics", [])
    drifted_features = []
    total_features = len(available)
    drifted_count = 0

    for metric in drift_metrics:
        if metric.get("metric") == "DatasetDriftMetric":
            data = metric.get("result", {})
            drifted_count = data.get("number_of_drifted_columns", 0)
            drift_share = data.get("share_of_drifted_columns", 0.0)
        elif metric.get("metric") == "ColumnDriftMetric":
            data = metric.get("result", {})
            if data.get("drift_detected"):
                col = data.get("column_name", "")
                drifted_features.append({
                    "feature": col,
                    "stattest": data.get("stattest_name", ""),
                    "drift_score": data.get("drift_score", 0.0),
                })

    drift_score = drifted_count / max(total_features, 1)

    # Save HTML report
    if output_path:
        report.save_html(output_path)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = DRIFT_REPORTS_DIR / f"drift_{timestamp}.html"
        report.save_html(str(report_path))
        logger.info(f"Drift report saved: {report_path}")

    return {
        "drift_detected": drift_score > 0.1,  # > 10% features drifted
        "drift_score": round(drift_score, 4),
        "drifted_features": drifted_features,
        "total_features_checked": total_features,
        "drifted_count": drifted_count,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _manual_psi_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """
    Fallback drift detection using Population Stability Index (PSI).
    PSI < 0.1: No drift | 0.1-0.2: Moderate | > 0.2: Significant
    """
    drifted_features = []
    available = [c for c in feature_columns if c in reference_df.columns and c in current_df.columns]

    for col in available:
        psi = _compute_psi(
            reference_df[col].dropna().values,
            current_df[col].dropna().values,
        )
        if psi > 0.2:
            drifted_features.append({"feature": col, "psi": round(psi, 4)})

    drift_score = len(drifted_features) / max(len(available), 1)
    return {
        "drift_detected": drift_score > 0.1,
        "drift_score": round(drift_score, 4),
        "drifted_features": drifted_features,
        "method": "PSI",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index."""
    eps = 1e-10
    ref_hist, bin_edges = np.histogram(reference, bins=n_bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_hist + eps) / (len(reference) + eps)
    cur_pct = (cur_hist + eps) / (len(current) + eps)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def trigger_retraining_if_drifted(
    drift_result: dict,
    drift_threshold: float,
    train_data_path: str,
    tracking_uri: str = "./mlflow_runs",
) -> Optional[str]:
    """
    If drift score exceeds threshold, trigger a retraining job.
    Logs the new model version to MLflow.

    Returns:
        New MLflow run ID if retrained, else None.
    """
    if not drift_result.get("drift_detected") or drift_result.get("drift_score", 0) < drift_threshold:
        logger.info(
            f"Drift score {drift_result.get('drift_score', 0):.3f} < threshold {drift_threshold}. "
            "No retraining triggered."
        )
        return None

    logger.warning(
        f"⚠️ Drift detected! Score: {drift_result.get('drift_score', 0):.3f} "
        f"(threshold: {drift_threshold}). Triggering retraining..."
    )

    try:
        from features.tabular import load_ibm_aml, engineer_features, get_feature_matrix
        from models.baseline import train_baseline

        df = load_ibm_aml(train_data_path)
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, scaler = get_feature_matrix(df)

        _, metrics = train_baseline(
            X_train, y_train, X_test, y_test,
            experiment_name="FraudGraph-Drift-Retrain",
            tracking_uri=tracking_uri,
        )

        # Save updated scaler
        scaler_path = Path("models/saved/scaler.pkl")
        scaler_path.parent.mkdir(exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info(
            f"Retraining complete. PR-AUC: {metrics.get('pr_auc', 'N/A')}"
        )

        # Log drift event to MLflow
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_name="drift_triggered_retrain") as run:
            mlflow.log_metrics({"drift_score": drift_result.get("drift_score", 0)})
            mlflow.log_metrics({f"retrain_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("triggered_by", "drift_monitor")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return None
