"""
Subgraph visualization for graph-based fraud explanations.
Shows the local transaction network around a flagged account.
This is the "relationship-level" explainability that SHAP can't provide.
"""

import logging
from typing import Optional

import networkx as nx
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def visualize_fraud_subgraph(
    G: nx.DiGraph,
    account_id: str,
    hops: int = 2,
    flagged_accounts: set | None = None,
) -> go.Figure:
    """
    Create an interactive Plotly visualization of the local fraud subgraph.

    Args:
        G: Full transaction graph (NetworkX DiGraph)
        account_id: The flagged account to center the visualization on
        hops: Number of hops to expand the subgraph
        flagged_accounts: Set of account IDs also flagged as suspicious

    Returns:
        Plotly Figure object (renderable in Streamlit with st.plotly_chart)
    """
    from features.graph_builder import get_fraud_subgraph

    if account_id not in G.nodes:
        logger.warning(f"Account {account_id} not found in graph.")
        # Return an empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Account {account_id} not in transaction graph",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16),
        )
        return fig

    subgraph = get_fraud_subgraph(G, account_id, hops=hops)
    if len(subgraph.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Empty subgraph", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Layout
    pos = nx.spring_layout(subgraph, seed=42, k=2)

    # ── Edge traces ───────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    edge_hover = []
    edge_colors = []

    for src, dst, data in subgraph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        is_fraud_edge = data.get("is_fraud", 0)
        edge_colors.append("#ef4444" if is_fraud_edge else "#64748b")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#64748b"),
        hoverinfo="none",
        name="Transactions",
    )

    # Fraud edges (highlighted)
    fraud_edge_x, fraud_edge_y = [], []
    for src, dst, data in subgraph.edges(data=True):
        if data.get("is_fraud", 0):
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])

    fraud_edge_trace = go.Scatter(
        x=fraud_edge_x, y=fraud_edge_y,
        mode="lines",
        line=dict(width=2.5, color="#ef4444"),
        hoverinfo="none",
        name="Fraud Transactions",
    )

    # ── Node traces ───────────────────────────────────────────────────────────
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    flagged_accounts = flagged_accounts or set()

    for node in subgraph.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Compute node stats
        out_degree = subgraph.out_degree(node)
        in_degree = subgraph.in_degree(node)
        fraud_edges_out = sum(1 for _, _, d in subgraph.out_edges(node, data=True) if d.get("is_fraud"))

        node_text.append(
            f"Account: {node[:12]}...<br>"
            f"Transactions sent: {out_degree}<br>"
            f"Transactions received: {in_degree}<br>"
            f"Fraud-flagged outgoing: {fraud_edges_out}"
        )

        if node == account_id:
            node_color.append("#f59e0b")  # amber — primary flagged account
            node_size.append(20)
        elif node in flagged_accounts:
            node_color.append("#ef4444")  # red — also flagged
            node_size.append(16)
        elif fraud_edges_out > 0:
            node_color.append("#f97316")  # orange — connected to fraud
            node_size.append(14)
        else:
            node_color.append("#3b82f6")  # blue — normal
            node_size.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="#1e293b"),
        ),
        name="Accounts",
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fraud_count = sum(1 for _, _, d in subgraph.edges(data=True) if d.get("is_fraud"))

    fig = go.Figure(
        data=[edge_trace, fraud_edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Transaction Network — Account {account_id[:16]}... ({hops}-hop subgraph)",
                font=dict(size=14, color="#f8fafc"),
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#f8fafc"),
            showlegend=True,
            hovermode="closest",
            annotations=[
                dict(
                    text=(
                        f"🔴 {fraud_count} fraud transactions | "
                        f"📊 {len(subgraph.nodes)} accounts | "
                        f"🔗 {len(subgraph.edges)} transactions"
                    ),
                    xref="paper", yref="paper",
                    x=0.01, y=-0.05,
                    showarrow=False,
                    font=dict(size=11, color="#94a3b8"),
                )
            ],
            legend=dict(
                bgcolor="#1e293b",
                bordercolor="#334155",
                font=dict(color="#f8fafc"),
            ),
            margin=dict(l=20, r=20, t=50, b=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig
