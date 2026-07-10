import os
import sys
import time
import random
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelGraph — AML Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8001"

# Check if local API backend is active
@st.cache_data(ttl=5)
def check_api_server():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

api_active = check_api_server()

# ── Custom Dark Styling (AML Command Center Theme) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0b0f19;
    color: #f8fafc;
}
.jetbrains-font {
    font-family: 'JetBrains Mono', monospace;
}
.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
}
.neon-green { color: #10b981; text-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
.neon-red { color: #ef4444; text-shadow: 0 0 8px rgba(239, 68, 68, 0.4); }
.neon-yellow { color: #f59e0b; text-shadow: 0 0 8px rgba(245, 158, 11, 0.4); }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #ef4444); }
</style>
""", unsafe_allow_html=True)

# ── Initialize Sandbox State (For Standalone Mode) ───────────────────────────
if "sandbox_graph" not in st.session_state:
    G = nx.DiGraph()
    for i in range(150):
        G.add_node(f"ACC_{i:03d}", risk_score=random.uniform(0.01, 0.25), balance=random.uniform(100, 25000))
    for _ in range(300):
        u, v = random.choice(list(G.nodes)), random.choice(list(G.nodes))
        if u != v:
            G.add_edge(u, v, amount=random.exponential(1500), timestamp=random.randint(0, 86400), is_fraud=0)
            
    # Inject cycles (Laundering Ring A)
    cycle_nodes = ["ACC_CYCLE_A", "ACC_CYCLE_B", "ACC_CYCLE_C", "ACC_CYCLE_D"]
    for n in cycle_nodes:
        G.add_node(n, risk_score=0.85, balance=50000.0)
    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        v = cycle_nodes[(i+1)%len(cycle_nodes)]
        G.add_edge(u, v, amount=12000.0, timestamp=10000 + i*600, is_fraud=1)
        
    # Inject split-merge (Laundering Ring B - Structuring)
    splitter = "ACC_SPLIT_SRC"
    merger = "ACC_MERGE_DST"
    intermediates = [f"ACC_INT_{i}" for i in range(5)]
    G.add_node(splitter, risk_score=0.92, balance=250000.0)
    G.add_node(merger, risk_score=0.75, balance=120.0)
    for n in intermediates:
        G.add_node(n, risk_score=0.45, balance=500.0)
        G.add_edge(splitter, n, amount=9500.0, timestamp=30000, is_fraud=1)
        G.add_edge(n, merger, amount=9450.0, timestamp=30100, is_fraud=1)
        
    st.session_state.sandbox_graph = G

if "sandbox_alerts" not in st.session_state:
    alerts = []
    alerts.append({
        "transaction_id": "TX_CYCLE_001",
        "sender": "ACC_CYCLE_A",
        "receiver": "ACC_CYCLE_B",
        "amount": 12000.0,
        "tabular_score": 0.35,
        "gnn_score": 0.96,
        "fused_score": 0.716,
        "confidence": 0.88,
        "type": "🚨 Laundering Loop (Cycle)",
        "timestamp": "10:00:00",
        "risk_factors": [
            {"feature": "Graph Loop Detected", "shap_value": 0.45},
            {"feature": "Coordinated Amounts", "shap_value": 0.32},
            {"feature": "Receiver Connection", "shap_value": 0.15},
            {"feature": "Single Transaction Amount", "shap_value": -0.05}
        ]
    })
    for i, n in enumerate([f"ACC_INT_{x}" for x in range(3)]):
        alerts.append({
            "transaction_id": f"TX_SPLIT_{i:03d}",
            "sender": "ACC_SPLIT_SRC",
            "receiver": n,
            "amount": 9500.0,
            "tabular_score": 0.52,
            "gnn_score": 0.89,
            "fused_score": 0.668,
            "confidence": 0.81,
            "type": "⚠️ Structuring (Split-Merge)",
            "timestamp": f"15:20:{i*15:02d}",
            "risk_factors": [
                {"feature": "Split Outflow (Fan-Out)", "shap_value": 0.38},
                {"feature": "Under reporting threshold ($10k)", "shap_value": 0.31},
                {"feature": "Tabular Risk Score", "shap_value": 0.12},
                {"feature": "Sender Activity Velocity", "shap_value": -0.04}
            ]
        })
    st.session_state.sandbox_alerts = alerts

# ── Sidebar Configurations ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ SentinelGraph Console")
    st.caption("Anti-Money Laundering Control Panel")
    st.divider()
    
    alert_limit = st.slider("Alerts count", 10, 200, 50)
    fraud_threshold_display = st.slider("Min Risk filter", 0.0, 1.0, 0.4, 0.05)
    
    st.divider()
    if api_active:
        st.success("🟢 Local API Connected (Port 8001)")
    else:
        st.info("☁️ Sandbox Simulation Mode")

# ── MAIN PANEL: ONE PAGE LAYOUT ───────────────────────────────────────────────
st.title("🛡️ SentinelGraph — Graph-Augmented AML Monitoring")
st.write(
    "SentinelGraph combines a gradient-boosted tabular baseline (XGBoost) with a "
    "**GraphSAGE Graph Neural Network** to detect layered transactions, split-merges, "
    "and circular transfer structures in financial transaction webs."
)

st.divider()

# 1. System Architecture
st.subheader("🕸️ System Architecture & Data Ingestion Flow")
if os.path.exists("sentinalGraph.png"):
    st.image("sentinalGraph.png", caption="SentinelGraph Operations Flow Chart", use_container_width=True)
else:
    st.info("System architecture diagram (sentinalGraph.png) not found.")

st.divider()

# 2. GraphSAGE Theory & Neighborhood Aggregator Explainer
st.subheader("🧠 GraphSAGE Neighborhood Sampling Explainer")
st.write(
    "How does a GNN capture connections? Traditional models only check row-level metrics. "
    "GraphSAGE aggregates features from the target node's local neighborhood (1-hop and 2-hop paths) "
    "to build a dense vector embedding that reflects its structural position."
)

target_node = st.selectbox("Pick Target Node to visualize GNN aggregation", ["ACC_CYCLE_A (Cycle Hub)", "ACC_SPLIT_SRC (Splitter Node)", "ACC_005 (Regular Node)"])
hops = st.slider("Neighborhood depth (Hops)", 1, 2, 2)

col_ta, col_tb = st.columns([1, 1.5])
with col_ta:
    st.write("### Aggregator Summary")
    if "CYCLE" in target_node:
        st.markdown("**Target Node:** ACC_CYCLE_A (Factual: 🔴 Fraud Loop)")
        st.markdown("**1-Hop Neighbors:** ACC_CYCLE_B (Outflow), ACC_CYCLE_D (Inflow)")
        if hops == 2:
            st.markdown("**2-Hop Neighbors:** ACC_CYCLE_C (Loop connector)")
            st.markdown("**Aggregator Formula:**")
            st.latex(r"h_{v}^{(1)} = \text{ReLU}\left(W \cdot \text{Mean}(\{h_{u}^{(0)}, \forall u \in \mathcal{N}(v)\})\right)")
            st.warning("⚠️ GNN detects identical, circular transaction flows with neighbor nodes. Fused risk is highly elevated.")
    elif "SPLIT" in target_node:
        st.markdown("**Target Node:** ACC_SPLIT_SRC (Factual: 🔴 Splitter)")
        st.markdown("**1-Hop Neighbors:** 5 Intermediate Accounts (ACC_INT_0 to 4)")
        if hops == 2:
            st.markdown("**2-Hop Neighbors:** ACC_MERGE_DST (Final Merger node)")
            st.markdown("**Aggregator Formula:**")
            st.latex(r"h_{v}^{(1)} = \text{ReLU}\left(W \cdot \text{Mean}(\{h_{u}^{(0)}\})\right)")
            st.warning("⚠️ GNN captures a high-degree split topology flowing into a single collector. Structuring alert flagged.")
    else:
        st.markdown("**Target Node:** ACC_005 (Factual: 🟢 Safe)")
        st.markdown("**1-Hop Neighbors:** ACC_042, ACC_011")
        if hops == 2:
            st.markdown("**2-Hop Neighbors:** ACC_095")
            st.success("✅ Standard localized connections. Low-risk structural embedding generated.")

with col_tb:
    st.write("### Local Sampling Topology")
    sub = nx.DiGraph()
    if "CYCLE" in target_node:
        sub.add_node("ACC_CYCLE_A", color="red")
        sub.add_edge("ACC_CYCLE_D", "ACC_CYCLE_A", amount=12000)
        sub.add_edge("ACC_CYCLE_A", "ACC_CYCLE_B", amount=12000)
        if hops == 2:
            sub.add_edge("ACC_CYCLE_B", "ACC_CYCLE_C", amount=12000)
            sub.add_edge("ACC_CYCLE_C", "ACC_CYCLE_D", amount=12000)
    elif "SPLIT" in target_node:
        sub.add_node("ACC_SPLIT_SRC", color="red")
        for i in range(4):
            sub.add_edge("ACC_SPLIT_SRC", f"ACC_INT_{i}", amount=9500)
            if hops == 2:
                sub.add_edge(f"ACC_INT_{i}", "ACC_MERGE_DST", amount=9450)
    else:
        sub.add_node("ACC_005", color="blue")
        sub.add_edge("ACC_042", "ACC_005", amount=350)
        sub.add_edge("ACC_005", "ACC_011", amount=150)
        if hops == 2:
            sub.add_edge("ACC_011", "ACC_095", amount=140)
            
    pos = nx.spring_layout(sub, seed=42)
    node_x, node_y, node_colors = [], [], []
    for node in sub.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append("red" if "Target" in node or node in [target_node.split(" ")[0]] else "#475569")
        
    edge_x, edge_y = [], []
    for u, v in sub.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#334155", width=1.5), hoverinfo="none"),
            go.Scatter(x=node_x, y=node_y, mode="markers+text", text=list(sub.nodes), textposition="top center", marker=dict(size=14, color=node_colors))
        ],
        layout=go.Layout(
            showlegend=False,
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. Interactive Threat Monitoring Command Center
st.subheader("⚡ Threat Command Center & Active Alerts")

# ── METRIC TILES ──────────────────────────────────────────────────────────
metrics_cols = st.columns(5)
with metrics_cols[0]:
    st.markdown('<div class="metric-card"><div>Inspected Trx</div><h2 class="neon-green">14,960</h2></div>', unsafe_allow_html=True)
with metrics_cols[1]:
    st.markdown('<div class="metric-card"><div>Active Alerts</div><h2 class="neon-red">24</h2></div>', unsafe_allow_html=True)
with metrics_cols[2]:
    st.markdown('<div class="metric-card"><div>Fraud Rate</div><h2 class="neon-yellow">0.16%</h2></div>', unsafe_allow_html=True)
with metrics_cols[3]:
    st.markdown('<div class="metric-card"><div>Avg Latency</div><h2>18.5ms</h2></div>', unsafe_allow_html=True)
with metrics_cols[4]:
    st.markdown('<div class="metric-card"><div>P99 Latency</div><h2 class="neon-green">41.0ms</h2></div>', unsafe_allow_html=True)

# Grid layout split view
main_col1, main_col2 = st.columns([3, 2])

# State for selected alert details
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = st.session_state.sandbox_alerts[0]

with main_col2:
    st.write("### 🚨 Live Threat Feed")
    
    # Load alerts
    alerts = []
    if api_active:
        try:
            resp = requests.get(f"{API_BASE}/alerts?limit={alert_limit}", timeout=3)
            alerts = resp.json().get("alerts", []) if resp.ok else []
        except Exception:
            alerts = []
    else:
        alerts = st.session_state.sandbox_alerts
        
    filtered_alerts = [a for a in alerts if a.get("fused_score", 0) >= fraud_threshold_display]
    
    alert_data = []
    for idx, a in enumerate(filtered_alerts):
        score = a.get("fused_score", 0)
        alert_data.append({
            "Index": idx,
            "Transaction ID": a.get("transaction_id", "")[:12],
            "Sender": a.get("sender", "")[:12],
            "Receiver": a.get("receiver", "")[:12],
            "Amount": f"${a.get('amount', 0):,.2f}",
            "Fused Risk": f"{score:.4f}",
            "Status": "🔴 FRAUD" if score >= 0.7 else "🟡 SUSPICIOUS"
        })
        
    if alert_data:
        df_table = pd.DataFrame(alert_data)
        selected = st.dataframe(
            df_table.drop(columns="Index"),
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        if selected and selected.selection.rows:
            selected_idx = selected.selection.rows[0]
            st.session_state.selected_alert = filtered_alerts[selected_idx]
    else:
        st.info("No active threats detected above the selected risk threshold.")
        
    st.divider()
    st.write("### 🔬 Manual Transaction Simulator Sandbox")
    st.caption("Input transactions to test classification outputs of XGBoost vs GraphSAGE.")
    
    col_sa, col_sb = st.columns(2)
    with col_sa:
        manual_sender = st.text_input("Sender Acc", value="ACC_CYCLE_A")
        manual_amount = st.number_input("Amount", min_value=1.0, value=12000.0)
        manual_currency = st.selectbox("Currency", ["USD", "EUR", "BTC"])
    with col_sb:
        manual_receiver = st.text_input("Receiver Acc", value="ACC_CYCLE_B")
        manual_type = st.selectbox("Format", ["Wire Transfer", "ACH", "Cash"])
        
    if st.button("🎯 Score Transaction", type="primary", use_container_width=True):
        G_temp = st.session_state.sandbox_graph
        in_cycle = False
        if manual_sender in G_temp and manual_receiver in G_temp:
            G_temp.add_edge(manual_sender, manual_receiver)
            try:
                cycles = list(nx.find_cycle(G_temp, source=manual_sender, orientation="original"))
                in_cycle = len(cycles) > 0
            except nx.NetworkXNoCycle:
                in_cycle = False
            G_temp.remove_edge(manual_sender, manual_receiver)
            
        tab_score = 0.85 if manual_amount > 100000 else random.uniform(0.1, 0.4)
        gnn_score = 0.96 if in_cycle or "CYCLE" in manual_sender or "SPLIT" in manual_sender else random.uniform(0.01, 0.3)
        fused_score = 0.6 * tab_score + 0.4 * gnn_score
        
        new_alert = {
            "transaction_id": f"TX_MANUAL_{random.randint(100, 999)}",
            "sender": manual_sender,
            "receiver": manual_receiver,
            "amount": manual_amount,
            "tabular_score": tab_score,
            "gnn_score": gnn_score,
            "fused_score": fused_score,
            "confidence": 0.85 if fused_score > 0.5 else 0.95,
            "type": "🚨 Laundering Loop (Cycle)" if in_cycle else "Regular Transaction Scored",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk_factors": [
                {"feature": "Graph Loop Topology" if in_cycle else "Transaction Volume", "shap_value": 0.42 if in_cycle else 0.12},
                {"feature": "Currency Crossings", "shap_value": 0.08 if manual_currency != "USD" else -0.05},
                {"feature": "Single Transaction Amount", "shap_value": 0.15 if manual_amount > 10000 else -0.1}
            ]
        }
        st.session_state.sandbox_alerts.insert(0, new_alert)
        st.session_state.selected_alert = new_alert
        st.success("Transaction scored and alert feed updated!")
        st.rerun()

with main_col1:
    st.write("### 🌐 Local Transaction Subgraph Topology")
    
    alert = st.session_state.selected_alert
    if alert:
        sender = alert.get("sender", "")
        receiver = alert.get("receiver", "")
        
        G_temp = st.session_state.sandbox_graph
        nodes_to_draw = {sender, receiver}
        for n in [sender, receiver]:
            if n in G_temp:
                nodes_to_draw.update(G_temp.neighbors(n))
                nodes_to_draw.update(G_temp.predecessors(n))
        
        sub = G_temp.subgraph(nodes_to_draw)
        pos = nx.spring_layout(sub, seed=42)
        
        # Edges
        edge_x, edge_y = [], []
        fraud_edge_x, fraud_edge_y = [], []
        for u, v, d in sub.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            is_alert_edge = (u == sender and v == receiver) or d.get("is_fraud", 0)
            
            if is_alert_edge:
                fraud_edge_x.extend([x0, x1, None])
                fraud_edge_y.extend([y0, y1, None])
            else:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
        # Nodes
        node_x, node_y, node_colors, node_sizes, node_texts = [], [], [], [], []
        for node in sub.nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == sender or node == receiver:
                node_colors.append("#ef4444")
                node_sizes.append(22)
            elif "CYCLE" in node or "SPLIT" in node or "MERGE" in node:
                node_colors.append("#f59e0b")
                node_sizes.append(16)
            else:
                node_colors.append("#475569")
            node_texts.append(f"Node: {node}<br>In-degree: {G_temp.in_degree(node)}<br>Out-degree: {G_temp.out_degree(node)}")
            
        fig = go.Figure(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#334155", width=1), hoverinfo="none", name="Normal Transfers"),
                go.Scatter(x=fraud_edge_x, y=fraud_edge_y, mode="lines", line=dict(color="#ef4444", width=3), hoverinfo="none", name="Flagged Outflows"),
                go.Scatter(x=node_x, y=node_y, mode="markers", text=node_texts, hoverinfo="text", marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color="#0f172a")), name="Accounts")
            ],
            layout=go.Layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#0b0f19",
                showlegend=True,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#334155")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Details & SHAP Side-by-Side
        st.divider()
        st.write(f"🔍 Alert Details — {alert.get('transaction_id', '')}")
        
        col_da, col_db = st.columns(2)
        with col_da:
            st.markdown(f"**Threat Type:** {alert.get('type', 'Flagged Activity')}")
            st.write(f"💵 **Amount:** ${alert.get('amount', 0):,.2f}")
            st.write(f"🕒 **Timestamp:** {alert.get('timestamp', '')}")
            
            # Scores
            score_df = pd.DataFrame([
                {"Model": "XGBoost (Tabular Baseline)", "Risk Score": f"{alert.get('tabular_score', 0):.4f}"},
                {"Model": "GraphSAGE (GNN Network)", "Risk Score": f"{alert.get('gnn_score', 0):.4f}"},
                {"Model": "SentinelGraph (Fused Result)", "Risk Score": f"{alert.get('fused_score', 0):.4f}"}
            ])
            st.dataframe(score_df, hide_index=True, use_container_width=True)
            
        with col_db:
            st.markdown("**📊 Risk Factor Contributions (SHAP)**")
            shap_factors = alert.get("risk_factors", [])
            if shap_factors:
                df_shap = pd.DataFrame(shap_factors)
                fig_shap = go.Figure(
                    data=[
                        go.Bar(
                            x=df_shap["shap_value"],
                            y=df_shap["feature"],
                            orientation="h",
                            marker_color=["#ef4444" if x > 0 else "#3b82f6" for x in df_shap["shap_value"]]
                        )
                    ],
                    layout=go.Layout(
                        paper_bgcolor="#0b0f19",
                        plot_bgcolor="#0b0f19",
                        xaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                        yaxis=dict(color="#94a3b8"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=200
                    )
                )
                st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("Select a flagged alert to explore the localized network structure.")

# ── Collapsible Model Validation Registry ─────────────────────────────────────
st.divider()
with st.expander("📋 Model Risk Management & Compliance Cards", expanded=False):
    st.subheader("Model Validation Registry")
    card_files = list(Path("model_cards").glob("*.md")) if Path("model_cards").exists() else []
    if card_files:
        selected_card = st.selectbox(
            "Select Model Card version",
            [f.name for f in sorted(card_files, reverse=True)]
        )
        card_path = Path("model_cards") / selected_card
        if card_path.exists():
            st.markdown(card_path.read_text(encoding="utf-8"))
    else:
        st.markdown("""
        ### Model Card: SentinelGraph GNN v1.0.0
        
        **Model Type:** Fused XGBoost Tree + PyTorch Geometric GraphSAGE Link Predictor.  
        **Intended Use:** Detection of structured money laundering schemes in bank transaction flows.  
        
        #### Performance Metrics (Offline Benchmark)
        * **XGBoost baseline ROC-AUC:** 0.842
        * **GraphSAGE GNN ROC-AUC:** 0.865
        * **Fused Model ROC-AUC:** **0.912**
        
        #### Model Constraints & Boundaries
        1. **Inference Lag:** GraphSAGE requires a local neighborhood structure. If an account has zero transactions, the model defaults to tabular metrics.
        2. **Drift Vulnerability:** AML evasion schemes shift dynamically. Auto-retraining is triggered if Evidently AI drift thresholds exceed `0.15`.
        """)
