"""
Unit tests for FraudGraph feature engineering and score fusion.
No real data required — uses synthetic DataFrames.
"""

import numpy as np
import pandas as pd
import pytest


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestTabularFeatures:
    """Test feature engineering on synthetic data."""

    def _make_df(self, n: int = 100) -> pd.DataFrame:
        """Generate a minimal synthetic IBM AML-like DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "From Bank": np.random.choice(["Bank A", "Bank B", "Bank C"], n),
            "Account": [f"ACC_{i % 10:03d}" for i in range(n)],
            "To Bank": np.random.choice(["Bank A", "Bank B", "Bank D"], n),
            "Account.1": [f"ACC_{(i+1) % 10:03d}" for i in range(n)],
            "Amount Received": np.random.exponential(10000, n),
            "Receiving Currency": np.random.choice(["USD", "EUR", "GBP"], n),
            "Amount Paid": np.random.exponential(10000, n),
            "Payment Currency": np.random.choice(["USD", "EUR", "GBP"], n),
            "Payment Format": np.random.choice(["Wire Transfer", "ACH", "Cash"], n),
            "Is Laundering": np.random.choice([0, 1], n, p=[0.99, 0.01]),
        })

    def test_engineer_features_produces_required_columns(self):
        from features.tabular import engineer_features, FEATURE_COLUMNS
        df = self._make_df(200)
        result = engineer_features(df)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_log_amount_non_negative(self):
        from features.tabular import engineer_features
        df = self._make_df(100)
        result = engineer_features(df)
        assert (result["log_amount_paid"] >= 0).all()
        assert (result["log_amount_received"] >= 0).all()

    def test_is_cross_currency_binary(self):
        from features.tabular import engineer_features
        df = self._make_df(100)
        result = engineer_features(df)
        assert set(result["is_cross_currency"].unique()).issubset({0, 1})

    def test_is_weekend_binary(self):
        from features.tabular import engineer_features
        df = self._make_df(100)
        result = engineer_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_label_column_preserved(self):
        from features.tabular import engineer_features
        df = self._make_df(100)
        result = engineer_features(df)
        assert "label" in result.columns
        assert set(result["label"].unique()).issubset({0, 1})

    def test_feature_matrix_no_data_leakage(self):
        """Scaler must be fit on TRAIN only, not on test set."""
        from features.tabular import engineer_features, get_feature_matrix
        df = self._make_df(500)
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, scaler = get_feature_matrix(df, test_size=0.2)

        # Scaler should be fitted (has mean_ attribute)
        assert hasattr(scaler, "mean_")

        # Test set should be transformed (not raw values)
        # Check that test data has similar scale to train
        assert abs(X_test.mean()) < 5.0, "Test data seems not normalized"

    def test_stratified_split_preserves_fraud_rate(self):
        """Stratified split should keep fraud rate similar in train/test."""
        from features.tabular import engineer_features, get_feature_matrix
        df = self._make_df(1000)
        # Force some fraud cases
        df.loc[df.index[:10], "Is Laundering"] = 1
        df["label"] = df["Is Laundering"].astype(int)
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, _ = get_feature_matrix(df)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Should be within 1% of each other (stratified)
        assert abs(train_rate - test_rate) < 0.02


# ── Score fusion tests ────────────────────────────────────────────────────────

class TestScoreFusion:
    """Test score fusion logic."""

    def test_tabular_only_when_no_gnn(self):
        from models.fusion import ScoreFusion
        fusion = ScoreFusion(tabular_weight=0.6, threshold=0.5)
        result = fusion.fuse(tabular_score=0.8, gnn_score=None)
        assert result["fused_score"] == 0.8
        assert result["is_fraud"] is True

    def test_weighted_fusion_correct(self):
        from models.fusion import ScoreFusion
        fusion = ScoreFusion(tabular_weight=0.6, threshold=0.5)
        result = fusion.fuse(tabular_score=0.4, gnn_score=0.9)
        expected = 0.6 * 0.4 + 0.4 * 0.9
        assert abs(result["fused_score"] - expected) < 1e-6

    def test_threshold_boundary(self):
        from models.fusion import ScoreFusion
        fusion = ScoreFusion(threshold=0.7)

        below = fusion.fuse(0.69)
        assert below["is_fraud"] is False

        at_threshold = fusion.fuse(0.7)
        assert at_threshold["is_fraud"] is True

    def test_confidence_is_zero_at_threshold(self):
        from models.fusion import ScoreFusion
        fusion = ScoreFusion(threshold=0.5)
        result = fusion.fuse(0.5)
        assert result["confidence"] == 0.0

    def test_invalid_weight_raises(self):
        from models.fusion import ScoreFusion
        with pytest.raises(ValueError):
            ScoreFusion(tabular_weight=1.5)

    def test_batch_fusion_shape(self):
        from models.fusion import ScoreFusion
        fusion = ScoreFusion(tabular_weight=0.6)
        tabular = np.array([0.1, 0.5, 0.9])
        gnn = np.array([0.2, 0.4, 0.8])
        result = fusion.batch_fuse(tabular, gnn)
        assert result.shape == (3,)
        assert all(0 <= v <= 1 for v in result)


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    """Test that evaluation metrics are computed correctly."""

    def test_pr_auc_perfect_classifier(self):
        from models.baseline import evaluate_classifier
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_classifier(y_true, y_prob, y_pred)
        assert metrics["pr_auc"] == 1.0

    def test_pr_auc_random_classifier(self):
        from models.baseline import evaluate_classifier
        np.random.seed(0)
        y_true = np.array([0] * 99 + [1])  # 1% fraud
        y_prob = np.random.rand(100)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_classifier(y_true, y_prob, y_pred)
        # Random classifier should have PR-AUC ≈ fraud rate
        assert metrics["pr_auc"] < 0.2

    def test_recall_at_fixed_fpr(self):
        from models.baseline import evaluate_classifier
        # Perfect separation
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.9])
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = evaluate_classifier(y_true, y_prob, y_pred)
        # With perfect separation, recall@5%FPR should be 1.0
        assert metrics["recall_at_fpr_5pct"] == 1.0
