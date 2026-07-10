"""
Feature engineering for IBM AML tabular data.
Produces features for the XGBoost/LightGBM baseline model.

Important: this dataset is severely imbalanced (<1% fraud).
We use SMOTE + class weighting — NEVER report plain accuracy.
Always report: Precision, Recall, PR-AUC, and recall@fixed-FPR.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# IBM AML column names
COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",            # sender account
    "To Bank",
    "Account.1",          # receiver account
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",      # label: 0 = legit, 1 = laundering
]


def load_ibm_aml(data_path: str) -> pd.DataFrame:
    """
    Load the IBM AML HI-Small dataset.

    Args:
        data_path: Path to HI-Small_Trans.csv

    Returns:
        Raw DataFrame with original columns + standardized label column.
    """
    logger.info(f"Loading IBM AML dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure label is integer
    if "Is Laundering" in df.columns:
        df["label"] = df["Is Laundering"].astype(int)
    elif "is_laundering" in df.columns:
        df["label"] = df["is_laundering"].astype(int)
    else:
        raise ValueError(f"Label column not found. Columns: {df.columns.tolist()}")

    fraud_rate = df["label"].mean() * 100
    logger.info(
        f"Loaded {len(df):,} transactions | "
        f"Fraud rate: {fraud_rate:.3f}% ({df['label'].sum():,} fraud cases)"
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer tabular features for fraud detection.

    Features created:
    - Time-based: hour, day of week, is_weekend
    - Amount features: log-amount, amount ratio (paid/received)
    - Currency: is_cross_currency (paid != received)
    - Sender/receiver aggregates: transaction count and total amount
      from same account in rolling window approximation
    - Payment format encoding
    - Cross-bank flag
    """
    logger.info("Engineering tabular features...")
    df = df.copy()

    # ── Parse timestamp ───────────────────────────────────────────────────────
    # IBM AML timestamps are in days since start OR datetime strings
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["hour"] = df["Timestamp"].dt.hour
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["day_of_month"] = df["Timestamp"].dt.day
    except Exception:
        # Numeric timestamp (days) — compute modular features
        df["hour"] = 0
        df["day_of_week"] = (df["Timestamp"].astype(float).astype(int) % 7)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["day_of_month"] = (df["Timestamp"].astype(float).astype(int) % 30) + 1

    # ── Amount features ───────────────────────────────────────────────────────
    df["amount_received"] = pd.to_numeric(df["Amount Received"], errors="coerce").fillna(0)
    df["amount_paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce").fillna(0)
    df["log_amount_paid"] = np.log1p(df["amount_paid"])
    df["log_amount_received"] = np.log1p(df["amount_received"])

    # Exchange rate / amount ratio — extreme values flag unusual patterns
    df["amount_ratio"] = np.where(
        df["amount_received"] > 0,
        df["amount_paid"] / df["amount_received"],
        1.0,
    )
    df["log_amount_ratio"] = np.log(np.clip(df["amount_ratio"], 0.01, 100))

    # ── Currency features ─────────────────────────────────────────────────────
    df["is_cross_currency"] = (
        df["Payment Currency"] != df["Receiving Currency"]
    ).astype(int)

    # Encode currencies
    le_pay_curr = LabelEncoder()
    le_recv_curr = LabelEncoder()
    all_currencies = pd.concat([df["Payment Currency"], df["Receiving Currency"]]).unique()
    le_pay_curr.fit(all_currencies)
    le_recv_curr.fit(all_currencies)
    df["payment_currency_enc"] = le_pay_curr.transform(df["Payment Currency"])
    df["receiving_currency_enc"] = le_recv_curr.transform(df["Receiving Currency"])

    # ── Payment format ────────────────────────────────────────────────────────
    le_format = LabelEncoder()
    df["payment_format_enc"] = le_format.fit_transform(df["Payment Format"].fillna("Unknown"))

    # ── Cross-bank flag ───────────────────────────────────────────────────────
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)

    # ── Sender/receiver aggregate features ───────────────────────────────────
    # Approximate rolling aggregates via global groupby stats
    # (In production, these would be computed from a Redis feature store)
    sender_stats = df.groupby("Account")["amount_paid"].agg(
        sender_tx_count="count",
        sender_total_amount="sum",
        sender_mean_amount="mean",
        sender_std_amount="std",
    ).fillna(0)

    receiver_stats = df.groupby("Account.1")["amount_received"].agg(
        receiver_tx_count="count",
        receiver_total_amount="sum",
        receiver_mean_amount="mean",
    ).fillna(0)

    df = df.join(sender_stats, on="Account")
    df = df.join(receiver_stats, on="Account.1")

    # How many unique receivers does this sender transact with?
    sender_diversity = df.groupby("Account")["Account.1"].nunique().rename("sender_receiver_diversity")
    df = df.join(sender_diversity, on="Account")

    # How many unique senders does this receiver receive from?
    receiver_diversity = df.groupby("Account.1")["Account"].nunique().rename("receiver_sender_diversity")
    df = df.join(receiver_diversity, on="Account.1")

    # Fill NaN from joins
    agg_cols = [
        "sender_tx_count", "sender_total_amount", "sender_mean_amount", "sender_std_amount",
        "receiver_tx_count", "receiver_total_amount", "receiver_mean_amount",
        "sender_receiver_diversity", "receiver_sender_diversity",
    ]
    for col in agg_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── Label column ──────────────────────────────────────────────────────────
    # Create/ensure 'label' column exists (int 0/1) for downstream use.
    # Works whether df came from load_ibm_aml (already has 'label') or
    # directly from a synthetic test DataFrame (has 'Is Laundering').
    if "label" not in df.columns:
        if "Is Laundering" in df.columns:
            df["label"] = df["Is Laundering"].astype(int)
        else:
            df["label"] = 0  # no label info available

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend", "day_of_month",
    "log_amount_paid", "log_amount_received", "log_amount_ratio",
    "is_cross_currency", "payment_currency_enc", "receiving_currency_enc",
    "payment_format_enc", "is_cross_bank",
    "sender_tx_count", "sender_total_amount", "sender_mean_amount", "sender_std_amount",
    "receiver_tx_count", "receiver_total_amount", "receiver_mean_amount",
    "sender_receiver_diversity", "receiver_sender_diversity",
]


def get_feature_matrix(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Extract feature matrix and split into train/test.
    Applies StandardScaler (fitted on train only — critical to avoid leakage).

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        logger.warning(f"Missing features (will skip): {missing}")

    X = df[available_features].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        f"Train: {len(X_train):,} ({y_train.sum():,} fraud) | "
        f"Test: {len(X_test):,} ({y_test.sum():,} fraud)"
    )
    return X_train, X_test, y_train, y_test, scaler
