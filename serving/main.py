"""
FastAPI scoring service for FraudGraph.
Endpoints:
  POST /score       — score a single transaction
  GET  /health      — health check
  GET  /metrics     — Prometheus-compatible metrics
  GET  /alerts      — recent high-risk transactions
"""

import asyncio
import logging
import pickle
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_account: str
    receiver_account: str
    amount_paid: float = Field(..., gt=0)
    amount_received: float = Field(..., gt=0)
    payment_currency: str = "USD"
    receiving_currency: str = "USD"
    payment_format: str = "Wire Transfer"
    sender_bank: str = ""
    receiver_bank: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ScoringResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fused_score: float
    tabular_score: float
    gnn_score: Optional[float]
    confidence: float
    top_risk_factors: list[dict]
    latency_ms: float
    model_version: str
    threshold: float


# ── In-memory stores ──────────────────────────────────────────────────────────
_recent_alerts: deque = deque(maxlen=500)  # ring buffer
_metrics_store = {
    "total_scored": 0,
    "total_flagged": 0,
    "total_errors": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_count": 0,
    "latency_p99_buffer": deque(maxlen=1000),
}

# ── Model state (loaded at startup) ──────────────────────────────────────────
_state = {
    "baseline_model": None,
    "scaler": None,
    "shap_explainer": None,
    "model_version": "0.0.0",
}


app = FastAPI(
    title="FraudGraph — Real-Time Fraud Scoring API",
    description=(
        "Hybrid XGBoost + GraphSAGE fraud/AML detection. "
        "POST a transaction, get a risk score + SHAP explanation in milliseconds."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load models at startup."""
    _load_models()


def _load_models():
    """Load baseline model and scaler from disk."""
    baseline_path = Path("models/saved/baseline_xgb.pkl")
    scaler_path = Path("models/saved/scaler.pkl")

    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            _state["baseline_model"] = pickle.load(f)
        logger.info("Loaded XGBoost baseline model.")
    else:
        logger.warning(
            "Baseline model not found. Run: python scripts/train.py first.\n"
            "API will return default scores until models are trained."
        )

    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            _state["scaler"] = pickle.load(f)
        logger.info("Loaded feature scaler.")

    # Initialize SHAP explainer if model is available
    if _state["baseline_model"] is not None:
        try:
            from features.tabular import FEATURE_COLUMNS
            from explainability.shap_explainer import SHAPExplainer
            _state["shap_explainer"] = SHAPExplainer(
                _state["baseline_model"], FEATURE_COLUMNS
            )
            logger.info("SHAP explainer initialized.")
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _state["baseline_model"] is not None,
        "model_version": _state["model_version"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics (simplified — use actual Prometheus client for production)."""
    store = _metrics_store
    n = store["latency_ms_count"]
    avg_latency = store["latency_ms_sum"] / max(n, 1)

    p99 = 0.0
    if store["latency_p99_buffer"]:
        buf = sorted(store["latency_p99_buffer"])
        p99_idx = int(len(buf) * 0.99)
        p99 = buf[min(p99_idx, len(buf) - 1)]

    return {
        "fraudgraph_transactions_total": store["total_scored"],
        "fraudgraph_flagged_total": store["total_flagged"],
        "fraudgraph_errors_total": store["total_errors"],
        "fraudgraph_latency_avg_ms": round(avg_latency, 2),
        "fraudgraph_latency_p99_ms": round(p99, 2),
        "fraudgraph_fraud_rate": round(
            store["total_flagged"] / max(store["total_scored"], 1), 6
        ),
    }


@app.post("/score", response_model=ScoringResponse)
async def score_transaction(transaction: TransactionRequest):
    """
    Score a single transaction for fraud.
    Returns risk score (0–1), binary flag, SHAP explanation, and latency.
    """
    t_start = time.perf_counter()

    try:
        result = _score_transaction_sync(transaction)

        latency_ms = (time.perf_counter() - t_start) * 1000
        result["latency_ms"] = round(latency_ms, 2)

        # Update metrics
        _metrics_store["total_scored"] += 1
        _metrics_store["latency_ms_sum"] += latency_ms
        _metrics_store["latency_ms_count"] += 1
        _metrics_store["latency_p99_buffer"].append(latency_ms)

        if result["is_fraud"]:
            _metrics_store["total_flagged"] += 1
            # Add to recent alerts
            _recent_alerts.append({
                **result,
                "transaction": transaction.dict(),
                "alerted_at": datetime.utcnow().isoformat() + "Z",
            })

        return ScoringResponse(**result)

    except Exception as e:
        _metrics_store["total_errors"] += 1
        logger.error(f"Scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Return the most recent flagged transactions."""
    alerts = list(_recent_alerts)[-limit:]
    return {
        "count": len(alerts),
        "alerts": list(reversed(alerts)),  # newest first
    }


# ── Core scoring logic ────────────────────────────────────────────────────────

def _score_transaction_sync(transaction: TransactionRequest) -> dict:
    """
    Core scoring logic — runs synchronously (called from async endpoint via thread).
    """
    from features.tabular import FEATURE_COLUMNS
    from models.fusion import ScoreFusion

    # ── Feature engineering (minimal, online version) ─────────────────────────
    dt = datetime.fromisoformat(transaction.timestamp.replace("Z", ""))
    features = _extract_online_features(transaction, dt)

    # ── Tabular scoring ───────────────────────────────────────────────────────
    tabular_score = 0.5  # default when model not loaded

    if _state["baseline_model"] is not None:
        feature_vec = np.array([features.get(f, 0.0) for f in FEATURE_COLUMNS], dtype=float)

        # Handle NaN
        feature_vec = np.nan_to_num(feature_vec, nan=0.0)

        if _state["scaler"] is not None:
            feature_vec = _state["scaler"].transform(feature_vec.reshape(1, -1))[0]

        tabular_score = float(
            _state["baseline_model"].predict_proba(feature_vec.reshape(1, -1))[0, 1]
        )

    # ── GNN scoring (None if not available at inference time) ─────────────────
    gnn_score = None  # In production: call GNN inference service

    # ── Score fusion ──────────────────────────────────────────────────────────
    fusion = ScoreFusion(
        tabular_weight=settings.tabular_model_weight,
        threshold=settings.fraud_threshold,
    )
    result = fusion.fuse(tabular_score, gnn_score)

    # ── SHAP explanation ──────────────────────────────────────────────────────
    top_risk_factors = []
    if _state["shap_explainer"] is not None and _state["baseline_model"] is not None:
        try:
            feature_vec = np.array([features.get(f, 0.0) for f in FEATURE_COLUMNS], dtype=float)
            feature_vec = np.nan_to_num(feature_vec, nan=0.0)
            if _state["scaler"] is not None:
                feature_vec_scaled = _state["scaler"].transform(feature_vec.reshape(1, -1))[0]
            else:
                feature_vec_scaled = feature_vec
            explanation = _state["shap_explainer"].explain_single(feature_vec_scaled)
            top_risk_factors = explanation.get("top_risk_factors", [])
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    return {
        "transaction_id": transaction.transaction_id,
        "is_fraud": result["is_fraud"],
        "fused_score": result["fused_score"],
        "tabular_score": result["tabular_score"],
        "gnn_score": result["gnn_score"],
        "confidence": result["confidence"],
        "top_risk_factors": top_risk_factors,
        "latency_ms": 0.0,  # filled by caller
        "model_version": _state["model_version"],
        "threshold": settings.fraud_threshold,
    }


def _extract_online_features(transaction: TransactionRequest, dt: datetime) -> dict:
    """Extract features for a single transaction at inference time."""

    amount_paid = transaction.amount_paid
    amount_received = transaction.amount_received
    ratio = amount_paid / max(amount_received, 1e-9)

    return {
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": int(dt.weekday() >= 5),
        "day_of_month": dt.day,
        "log_amount_paid": math.log1p(amount_paid),
        "log_amount_received": math.log1p(amount_received),
        "log_amount_ratio": math.log(max(min(ratio, 100), 0.01)),
        "is_cross_currency": int(transaction.payment_currency != transaction.receiving_currency),
        "payment_currency_enc": hash(transaction.payment_currency) % 100,
        "receiving_currency_enc": hash(transaction.receiving_currency) % 100,
        "payment_format_enc": hash(transaction.payment_format) % 50,
        "is_cross_bank": int(transaction.sender_bank != transaction.receiver_bank),
        # Rolling aggregates: in production, these come from Redis
        # Here we use neutral values (0) — explain this in the README
        "sender_tx_count": 0,
        "sender_total_amount": 0,
        "sender_mean_amount": 0,
        "sender_std_amount": 0,
        "receiver_tx_count": 0,
        "receiver_total_amount": 0,
        "receiver_mean_amount": 0,
        "sender_receiver_diversity": 0,
        "receiver_sender_diversity": 0,
    }
