"""
SHAP explainability for the XGBoost baseline model.
Produces feature-level explanations per flagged transaction.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based feature importance explainer for the tabular fraud model.
    Uses TreeExplainer (fast, exact for tree models).
    """

    def __init__(self, model, feature_names: list[str]):
        import shap
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized.")

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Compute SHAP values for a single transaction.

        Args:
            x: Feature vector (1D array, shape = n_features)

        Returns:
            Dict with: shap_values, feature_names, base_value, top_factors
        """
        import shap

        x_2d = x.reshape(1, -1) if x.ndim == 1 else x
        shap_values = self.explainer.shap_values(x_2d)

        # For binary classification, shap_values may be [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # fraud class
        else:
            sv = shap_values[0]

        # Build explanation dict
        feature_contributions = dict(zip(self.feature_names, sv.tolist()))

        # Top 5 factors by absolute contribution
        top_factors = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        return {
            "shap_values": dict(zip(self.feature_names, sv.tolist())),
            "feature_names": self.feature_names,
            "base_value": float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value),
            "top_risk_factors": [
                {
                    "feature": name,
                    "shap_value": round(val, 6),
                    "direction": "increases fraud risk" if val > 0 else "decreases fraud risk",
                }
                for name, val in top_factors
            ],
        }

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of transactions.

        Returns:
            SHAP values array, shape (n_samples, n_features)
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def plot_waterfall(
        self,
        x: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        """
        Generate a SHAP waterfall plot for a single transaction.

        Returns:
            Path to saved PNG, or None.
        """
        import shap

        x_2d = x.reshape(1, -1) if x.ndim == 1 else x
        shap_values = self.explainer(x_2d)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        if show:
            plt.show()
        plt.close()
        return None

    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 15,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Global feature importance summary plot."""
        import shap

        shap_values = self.explain_batch(X)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        plt.show()
        plt.close()
        return None
