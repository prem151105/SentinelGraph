"""
FraudGraph — Analyst Streamlit Dashboard
Shows:
  - Live-updating table of flagged transaction alerts
  - SHAP waterfall plot for each alert
  - Fraud subgraph visualization (who's connected to whom)
  - Model card for the version that produced the score
  - System metrics (latency, throughput, fraud rate)
"""

import json
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

API_BASE = "http://localhost:8001"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGraph — AML Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.fraud-badge {
    background: #7f1d1d; color: #fca5a5;
    padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem;
}
.safe-badge {
    background: #064e3b; color: #6ee7b7;
    padding: 3px 10px; border-radius: 999px; font-size: 0.8rem;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ FraudGraph")
    st.caption("Real-Time AML Detection System")
    st.divider()

    auto_refresh = st.checkbox("Auto-refresh alerts (5s)", value=True)
    alert_limit = st.slider("Alerts to show", 10, 200, 50)
    fraud_threshold_display = st.slider("Display threshold filter", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.caption("**Stack:**")
    st.caption("📊 XGBoost (tabular baseline)")
    st.caption("🕸️ GraphSAGE (graph GNN)")
    st.caption("⚗️ Score fusion")
    st.caption("🔍 SHAP explanations")
    st.caption("🌐 Subgraph visualization")

    st.divider()
    if st.button("🔄 Refresh Now"):
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛡️ FraudGraph — AML Detection Dashboard")
st.caption("Real-time fraud/money-laundering detection | XGBoost + GraphSAGE")

# ── System metrics row ────────────────────────────────────────────────────────
try:
    metrics_resp = requests.get(f"{API_BASE}/metrics", timeout=3)
    system_metrics = metrics_resp.json() if metrics_resp.ok else {}
except Exception:
    system_metrics = {}

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Transactions Scored", f"{system_metrics.get('fraudgraph_transactions_total', 0):,}")
with col2:
    st.metric("Fraud Alerts", f"{system_metrics.get('fraudgraph_flagged_total', 0):,}")
with col3:
    fraud_rate = system_metrics.get("fraudgraph_fraud_rate", 0)
    st.metric("Fraud Rate", f"{fraud_rate*100:.2f}%")
with col4:
    st.metric("Avg Latency", f"{system_metrics.get('fraudgraph_latency_avg_ms', 0):.1f}ms")
with col5:
    p99 = system_metrics.get("fraudgraph_latency_p99_ms", 0)
    color = "🟢" if p99 < 50 else "🟡" if p99 < 200 else "🔴"
    st.metric("P99 Latency", f"{color} {p99:.1f}ms")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🚨 Live Alerts", "🔬 Transaction Scorer", "📋 Model Card"
])

# ── Tab 1: Live Alerts ────────────────────────────────────────────────────────
with tab1:
    st.subheader("🚨 Recent Fraud Alerts")

    try:
        alerts_resp = requests.get(f"{API_BASE}/alerts?limit={alert_limit}", timeout=5)
        alerts_data = alerts_resp.json() if alerts_resp.ok else {"alerts": [], "count": 0}
    except Exception:
        alerts_data = {"alerts": [], "count": 0}

    alerts = alerts_data.get("alerts", [])
    filtered_alerts = [
        a for a in alerts
        if a.get("fused_score", 0) >= fraud_threshold_display
    ]

    if not alerts:
        st.info(
            "No alerts yet. Start the transaction simulator:\n"
            "```\npython serving/simulator.py ./data/raw/HI-Small_Trans.csv\n```"
        )
    else:
        st.caption(f"Showing {len(filtered_alerts)} alerts above {fraud_threshold_display:.0%} threshold")

        # Alert table
        table_data = []
        for a in filtered_alerts[:50]:
            tx = a.get("transaction", {})
            score = a.get("fused_score", 0)
            table_data.append({
                "Transaction ID": a.get("transaction_id", "")[:16] + "...",
                "Sender": tx.get("sender_account", "")[:12] + "...",
                "Receiver": tx.get("receiver_account", "")[:12] + "...",
                "Amount": f"${tx.get('amount_paid', 0):,.2f}",
                "Score": f"{score:.4f}",
                "Confidence": f"{a.get('confidence', 0):.2%}",
                "Status": "🔴 FRAUD" if score >= 0.7 else "🟡 SUSPICIOUS",
                "Alerted At": a.get("alerted_at", "")[:19],
            })

        if table_data:
            df_table = pd.DataFrame(table_data)
            selected = st.dataframe(
                df_table,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
            )

            # Detail view for selected alert
            if selected and selected.selection.rows:
                selected_idx = selected.selection.rows[0]
                alert = filtered_alerts[selected_idx]

                st.divider()
                st.subheader(f"🔍 Alert Detail — {alert.get('transaction_id', '')[:20]}")

                detail_col1, detail_col2 = st.columns([1, 2])

                with detail_col1:
                    st.markdown("**Risk Score Breakdown**")
                    tx = alert.get("transaction", {})
                    st.metric("Fused Score", f"{alert.get('fused_score', 0):.4f}")
                    st.metric("Tabular Score", f"{alert.get('tabular_score', 0):.4f}")
                    st.metric("GNN Score", f"{alert.get('gnn_score', 'N/A')}")
                    st.metric("Confidence", f"{alert.get('confidence', 0):.2%}")

                    st.markdown("**Transaction Details**")
                    st.json({
                        "sender": tx.get("sender_account", ""),
                        "receiver": tx.get("receiver_account", ""),
                        "amount": f"${tx.get('amount_paid', 0):,.2f}",
                        "currency": f"{tx.get('payment_currency', '')} → {tx.get('receiving_currency', '')}",
                        "format": tx.get("payment_format", ""),
                    })

                with detail_col2:
                    # SHAP explanation
                    st.markdown("**📊 Top Risk Factors (SHAP)**")
                    risk_factors = alert.get("top_risk_factors", [])
                    if risk_factors:
                        factors_df = pd.DataFrame(risk_factors)
                        if "shap_value" in factors_df.columns:
                            factors_df["Impact"] = factors_df["shap_value"].apply(
                                lambda x: "↑ Fraud Risk" if x > 0 else "↓ Fraud Risk"
                            )
                            st.dataframe(
                                factors_df[["feature", "shap_value", "Impact"]],
                                hide_index=True,
                                use_container_width=True,
                            )
                    else:
                        st.caption("SHAP explanation not available (model not loaded).")

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ── Tab 2: Manual Transaction Scorer ─────────────────────────────────────────
with tab2:
    st.subheader("🔬 Score a Transaction Manually")

    col_a, col_b = st.columns(2)
    with col_a:
        sender = st.text_input("Sender Account", value="ACC_001")
        amount_paid = st.number_input("Amount Paid ($)", min_value=1.0, value=50000.0, step=1000.0)
        payment_currency = st.selectbox("Payment Currency", ["USD", "EUR", "GBP", "BTC", "ETH"])
        payment_format = st.selectbox("Payment Format", ["Wire Transfer", "ACH", "Reinvestment", "Cash", "Cheque", "Credit Cards"])
        sender_bank = st.text_input("Sender Bank", value="Bank A")

    with col_b:
        receiver = st.text_input("Receiver Account", value="ACC_002")
        amount_received = st.number_input("Amount Received", min_value=1.0, value=49500.0, step=1000.0)
        receiving_currency = st.selectbox("Receiving Currency", ["USD", "EUR", "GBP", "BTC", "ETH"])
        receiver_bank = st.text_input("Receiver Bank", value="Bank B")

    if st.button("🎯 Score Transaction", type="primary"):
        payload = {
            "sender_account": sender,
            "receiver_account": receiver,
            "amount_paid": amount_paid,
            "amount_received": amount_received,
            "payment_currency": payment_currency,
            "receiving_currency": receiving_currency,
            "payment_format": payment_format,
            "sender_bank": sender_bank,
            "receiver_bank": receiver_bank,
        }

        try:
            resp = requests.post(f"{API_BASE}/score", json=payload, timeout=10)
            if resp.ok:
                result = resp.json()
                score = result.get("fused_score", 0)

                st.divider()
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    color = "🔴" if result["is_fraud"] else "🟢"
                    st.metric("Decision", f"{color} {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
                with res_col2:
                    st.metric("Risk Score", f"{score:.4f}")
                with res_col3:
                    st.metric("Latency", f"{result.get('latency_ms', 0):.1f}ms")

                # SHAP factors
                factors = result.get("top_risk_factors", [])
                if factors:
                    st.subheader("Top Risk Factors (SHAP)")
                    for f in factors:
                        direction = "🔺" if f.get("shap_value", 0) > 0 else "🔻"
                        st.caption(f"{direction} {f['feature']}: {f.get('shap_value', 0):.4f} ({f.get('direction', '')})")
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

# ── Tab 3: Model Card ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 Model Card")

    card_files = list(Path("model_cards").glob("*.md")) if Path("model_cards").exists() else []

    if card_files:
        selected_card = st.selectbox(
            "Select model version",
            [f.name for f in sorted(card_files, reverse=True)],
        )
        card_path = Path("model_cards") / selected_card
        if card_path.exists():
            st.markdown(card_path.read_text(encoding="utf-8"))
    else:
        st.info(
            "No model cards yet. Run training first:\n"
            "```\npython scripts/train.py --data ./data/raw/HI-Small_Trans.csv\n```"
        )
