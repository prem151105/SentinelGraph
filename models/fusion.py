"""
Score fusion module.
Combines XGBoost (tabular) and GraphSAGE (graph) scores
into a single risk score with a configurable decision threshold.

Fusion strategy: weighted average
  score_final = w * score_tabular + (1 - w) * score_gnn

The weight is configurable (default 60% tabular, 40% GNN).
This can be tuned to a meta-model (logistic regression over both scores)
if you have enough data — the README explains the tradeoff.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ScoreFusion:
    """
    Combines tabular and graph model scores into a single risk score.

    Args:
        tabular_weight: Weight for XGBoost score (0.0–1.0). GNN gets 1-w.
        threshold: Decision threshold for binary fraud classification.
    """

    def __init__(self, tabular_weight: float = 0.6, threshold: float = 0.5):
        if not 0 <= tabular_weight <= 1:
            raise ValueError(f"tabular_weight must be in [0, 1], got {tabular_weight}")
        self.tabular_weight = tabular_weight
        self.gnn_weight = 1 - tabular_weight
        self.threshold = threshold

    def fuse(
        self,
        tabular_score: float,
        gnn_score: Optional[float] = None,
    ) -> dict:
        """
        Compute the fused risk score.

        Args:
            tabular_score: XGBoost fraud probability (0–1)
            gnn_score: GraphSAGE fraud probability (0–1), or None if not available

        Returns:
            Dict with: fused_score, tabular_score, gnn_score, is_fraud, confidence
        """
        if gnn_score is None:
            # Fall back to tabular only (GNN not available for this transaction)
            fused = float(tabular_score)
            weights_used = {"tabular": 1.0, "gnn": 0.0}
        else:
            fused = (
                self.tabular_weight * float(tabular_score)
                + self.gnn_weight * float(gnn_score)
            )
            weights_used = {"tabular": self.tabular_weight, "gnn": self.gnn_weight}

        is_fraud = fused >= self.threshold

        # Confidence: distance from decision boundary, scaled to 0–1
        confidence = abs(fused - self.threshold) / max(self.threshold, 1 - self.threshold)

        return {
            "fused_score": round(fused, 6),
            "tabular_score": round(float(tabular_score), 6),
            "gnn_score": round(float(gnn_score), 6) if gnn_score is not None else None,
            "is_fraud": bool(is_fraud),
            "confidence": round(float(confidence), 4),
            "weights_used": weights_used,
            "threshold": self.threshold,
        }

    def batch_fuse(
        self,
        tabular_scores: np.ndarray,
        gnn_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse scores for a batch.

        Returns:
            Array of fused probabilities.
        """
        if gnn_scores is None:
            return tabular_scores.astype(float)
        return (
            self.tabular_weight * tabular_scores
            + self.gnn_weight * gnn_scores
        ).astype(float)

    def evaluate_fused(
        self,
        y_true: np.ndarray,
        tabular_scores: np.ndarray,
        gnn_scores: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Evaluate the fused model on a test set.

        Returns:
            Metrics dict (same format as baseline.evaluate_classifier)
        """
        from models.baseline import evaluate_classifier
        fused = self.batch_fuse(tabular_scores, gnn_scores)
        y_pred = (fused >= self.threshold).astype(int)
        return evaluate_classifier(y_true, fused, y_pred)
