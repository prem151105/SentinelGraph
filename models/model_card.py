"""
Auto-generate model cards per deployed model version.
Mirrors real bank model-risk-management practice.

A model card documents:
- What the model does and what it's intended for
- Training data summary
- Performance metrics
- Known limitations and blind spots
- Not-intended-use notes
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_model_card(
    model_name: str,
    model_version: str,
    mlflow_run_id: str,
    baseline_metrics: dict,
    gnn_metrics: dict | None,
    fused_metrics: dict | None,
    training_data_info: dict,
    output_dir: str = "./model_cards",
) -> str:
    """
    Generate a model card markdown file for a deployed model version.

    Args:
        model_name: Human-readable model name
        model_version: Semantic version (e.g. "1.0.0")
        mlflow_run_id: MLflow run ID for traceability
        baseline_metrics: XGBoost evaluation metrics
        gnn_metrics: GraphSAGE metrics (or None if not used)
        fused_metrics: Fused model metrics
        training_data_info: Dict with dataset info
        output_dir: Where to save the model card

    Returns:
        Path to the generated model card file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"model_card_{model_name.replace(' ', '_').lower()}_v{model_version}.md"
    output_path = Path(output_dir) / filename

    def fmt_metric(m: dict | None, key: str) -> str:
        if m is None:
            return "N/A"
        val = m.get(key)
        if val is None:
            return "N/A"
        return f"{val:.4f}"

    card = f"""# Model Card: {model_name}

**Version:** {model_version}  
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}  
**MLflow Run ID:** `{mlflow_run_id}`  

---

## Model Overview

**Task:** Binary fraud/AML transaction classification  
**Type:** Hybrid — XGBoost tabular baseline + GraphSAGE GNN (edge-level)  
**Score fusion:** Weighted average ({int(training_data_info.get('tabular_weight', 0.6)*100)}% tabular, {int((1-training_data_info.get('tabular_weight', 0.6))*100)}% GNN)  
**Decision threshold:** {training_data_info.get('threshold', 0.5)}  

**Intended use:**
- Flagging potentially fraudulent or money-laundering transactions for analyst review
- NOT for fully automated blocking decisions without human review
- NOT for deployment in highly regulated jurisdictions without additional compliance validation

**Not intended for:**
- Real-time blocking without a human-in-the-loop review step
- Detecting fraud types not present in the training data (see Known Limitations)
- Direct customer communication or adverse action decisions

---

## Training Data

| Property | Value |
|----------|-------|
| Primary dataset | IBM AML (HI-Small synthetic) |
| Secondary dataset | Elliptic Bitcoin graph (real-world validation) |
| Training period | {training_data_info.get('date_range', 'Full dataset')} |
| Training samples | {training_data_info.get('train_samples', 'N/A'):,} |
| Test samples | {training_data_info.get('test_samples', 'N/A'):,} |
| Fraud rate (train) | {training_data_info.get('fraud_rate_train', 0):.3%} |

**Data notes:**
- IBM AML dataset is **synthetic** — designed to simulate real laundering patterns but does not reflect actual transaction distributions from any specific institution
- Elliptic dataset is a **real** Bitcoin transaction graph (anonymized) — used as secondary validation
- Severe class imbalance (<1% fraud) is a feature, not a bug — it mirrors real-world conditions

---

## Performance Metrics

> **Why not accuracy?** With <1% fraud rate, a model predicting "all legitimate" achieves >99% accuracy — completely useless. We use PR-AUC, precision/recall, and recall@fixed-FPR.

### XGBoost Baseline (tabular features only)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **{fmt_metric(baseline_metrics, 'pr_auc')}** |
| ROC-AUC | {fmt_metric(baseline_metrics, 'roc_auc')} |
| Precision @0.5 threshold | {fmt_metric(baseline_metrics, 'precision_at_05')} |
| Recall @0.5 threshold | {fmt_metric(baseline_metrics, 'recall_at_05')} |
| F1 @0.5 threshold | {fmt_metric(baseline_metrics, 'f1_at_05')} |

### GraphSAGE GNN (graph features)
{_format_gnn_metrics(gnn_metrics)}

### Fused Model (tabular + graph)
{_format_fused_metrics(fused_metrics)}

---

## Known Limitations

1. **Synthetic training data:** The IBM AML dataset simulates laundering patterns but may not capture novel real-world evasion techniques. Performance may degrade on out-of-distribution fraud types.

2. **Graph freshness:** The GNN is trained on a snapshot of the transaction graph. In production, graph updates (new accounts, new edges) require periodic retraining to maintain accuracy.

3. **Cold start problem:** New accounts with no transaction history lack meaningful graph features. The system falls back to tabular-only scoring for these accounts.

4. **Concept drift:** Fraud patterns evolve over time. Evidently AI monitoring tracks distribution shift — see drift reports in `mlops/drift_reports/`.

5. **Adversarial adaptation:** Sophisticated fraud rings may adapt behavior to evade detection once they learn the model's decision boundary. Periodic retraining and threshold review are essential.

6. **Class imbalance:** Despite SMOTE and class weighting, the model may produce many false positives at high-recall settings. Threshold selection should be driven by the operational FPR budget.

7. **Explainability on graph side:** SHAP explanations are available for the tabular model. Graph explanations (subgraph visualization) are qualitative — they show connectivity, not a formal feature attribution score.

---

## Recommended Use

| Use Case | Recommendation |
|----------|----------------|
| Analyst review queue | ✅ Use fused score with configurable threshold |
| SARs (Suspicious Activity Reports) | ✅ Use as evidence alongside analyst judgment |
| Automated blocking | ⚠️ Requires additional validation and regulatory approval |
| Customer-facing adverse decisions | ❌ Not intended without additional FCRA/ECOA compliance review |

---

## Retraining Policy

- **Trigger:** Evidently AI drift score > {training_data_info.get('drift_threshold', 0.15)} on rolling 1,000-transaction window
- **Frequency:** Minimum quarterly review regardless of drift
- **Process:** Automated retraining pipeline → MLflow staging → human review → promotion to production
- **Version control:** Each retrain creates a new model version in MLflow registry

---

*This model card was auto-generated by FraudGraph v{model_version}. Review and update before production deployment.*
"""

    output_path.write_text(card, encoding="utf-8")
    logger.info(f"Model card written to: {output_path}")
    return str(output_path)


def _format_gnn_metrics(metrics: dict | None) -> str:
    if metrics is None:
        return "_GNN not trained for this model version._"

    def f(k):
        v = metrics.get(k)
        return f"{v:.4f}" if v is not None else "N/A"

    return f"""| Metric | Value |
|--------|-------|
| **PR-AUC** | **{f('pr_auc')}** |
| ROC-AUC | {f('roc_auc')} |
| Precision @0.5 | {f('precision_at_05')} |
| Recall @0.5 | {f('recall_at_05')} |
| F1 @0.5 | {f('f1_at_05')} |"""


def _format_fused_metrics(metrics: dict | None) -> str:
    if metrics is None:
        return "_Fused model metrics not available._"
    return _format_gnn_metrics(metrics)
