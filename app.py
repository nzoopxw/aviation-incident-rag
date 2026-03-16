import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.rag_chain import query_rag

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Incident Pattern Finder",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Aviation Incident Report Pattern Finder")
st.caption("Powered by NASA ASRS dataset (2022–2024) · Hybrid RAG + Semantic Clustering")

# ── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("Filters")

phase_options = ["All", "Landing", "Takeoff / Launch", "Cruise", "Initial Approach",
                 "Final Approach", "Climb", "Descent", "Taxi"]
selected_phase = st.sidebar.selectbox("Phase of Flight", phase_options)

aircraft_filter = st.sidebar.text_input("Aircraft Type (e.g. B737)", "")

st.sidebar.markdown("---")
st.sidebar.markdown("**Example queries:**")
st.sidebar.markdown("- hydraulic failure during landing")
st.sidebar.markdown("- TCAS resolution advisory ignored")
st.sidebar.markdown("- crew fatigue approach unstabilized")
st.sidebar.markdown("- runway incursion ATC communication")

# ── Query input ────────────────────────────────────────────────────────────
query = st.text_input("Enter your query:", placeholder="e.g. hydraulic failure during landing")

run = st.button("Analyze", type="primary")

if run and query:
    # Build filters
    filters = {}
    if selected_phase != "All":
        filters["phase_of_flight"] = selected_phase
    if aircraft_filter.strip():
        filters["aircraft"] = aircraft_filter.strip()

    with st.spinner("Retrieving incidents and detecting patterns..."):
        result = query_rag(query, filters=filters if filters else None)

    # ── Answer ─────────────────────────────────────────────────────────────
    st.markdown("## 🔍 Analysis")
    st.markdown(result["answer"])

    # ── Patterns ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Detected Patterns")

    col1, col2, col3 = st.columns(3)
    col1.metric("Reports Retrieved", result["patterns"]["total_retrieved"])
    col2.metric("Clusters Found", len(result["patterns"]["clusters"]))
    col3.metric("Reports Clustered", result["patterns"].get("total_clustered", 0))

    # Temporal chart
    temporal = result["patterns"].get("temporal", {})
    if temporal:
        st.markdown("### Incident Frequency Over Time")
        fig = px.bar(
            x=list(temporal.keys()),
            y=list(temporal.values()),
            labels={"x": "Year", "y": "Number of Incidents"},
            color_discrete_sequence=["#e63946"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Co-occurrence network
    co_occ = result["patterns"].get("co_occurrences", [])
    if co_occ:
        st.markdown("### Contributing Factor Co-occurrences")
        G = nx.Graph()
        for u, v, w in co_occ:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = list(G.nodes())

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                   line=dict(width=1, color="#888"), hoverinfo="none"))
        fig2.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                   marker=dict(size=20, color="#e63946"),
                                   text=node_text, textposition="top center"))
        fig2.update_layout(showlegend=False, height=400,
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster topics
    clusters = result["patterns"].get("clusters", {})
    if clusters:
        st.markdown("### Semantic Clusters")
        for label, info in clusters.items():
            with st.expander(f"Cluster {label} — {info['size']} incidents — Topics: {', '.join(info['keywords'][:3])}"):
                for chunk in info["chunks"]:
                    st.markdown(f"**ACN {chunk['acn']}** | {chunk['date']} | {chunk['aircraft']} | {chunk['phase_of_flight']}")
                    st.markdown(f"> {chunk['text'][:300]}...")
                    st.markdown("---")

    # Raw results
    with st.expander("📄 View retrieved incident reports"):
        for i, chunk in enumerate(result["retrieved_chunks"]):
            st.markdown(f"**[{i+1}] ACN {chunk['acn']}** | {chunk['date']} | {chunk['aircraft']} | {chunk['phase_of_flight']}")
            st.markdown(chunk["text"])
            st.markdown("---")