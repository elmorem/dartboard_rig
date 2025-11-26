"""
Streamlit Comparison Interface for Retrieval Methods.

This app provides a web interface to compare different retrieval methods
(BM25, Dense, Hybrid, Dartboard) side-by-side with visualization of results,
metrics, and overlap analysis.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any
import os

# Page configuration
st.set_page_config(
    page_title="Dartboard RAG - Retrieval Comparison",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-12345")


def call_compare_api(
    query: str, methods: List[str], top_k: int, use_reranker: bool
) -> Dict[str, Any]:
    """
    Call the /compare API endpoint.

    Args:
        query: Search query
        methods: List of retrieval methods to compare
        top_k: Number of chunks to retrieve
        use_reranker: Whether to apply cross-encoder reranking

    Returns:
        API response as dictionary
    """
    url = f"{API_BASE_URL}/compare"
    headers = {"X-API-Key": API_KEY}
    payload = {
        "query": query,
        "methods": methods,
        "top_k": top_k,
        "use_reranker": use_reranker,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            st.error(f"Response: {e.response.text}")
        return None


def create_latency_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """Create bar chart comparing retrieval latencies."""
    methods = [r["method"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    fig = go.Figure(data=[go.Bar(x=methods, y=latencies, marker_color="lightblue")])

    fig.update_layout(
        title="Retrieval Latency Comparison",
        xaxis_title="Method",
        yaxis_title="Latency (ms)",
        height=400,
    )

    return fig


def create_score_distribution_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """Create box plot comparing score distributions."""
    fig = go.Figure()

    for result in results:
        fig.add_trace(
            go.Box(
                y=result["scores"],
                name=result["method"],
                boxmean="sd",
            )
        )

    fig.update_layout(
        title="Score Distribution by Method",
        yaxis_title="Relevance Score",
        height=400,
    )

    return fig


def create_overlap_heatmap(overlap_matrix: Dict[str, Dict]) -> go.Figure:
    """Create heatmap showing overlap between methods."""
    if not overlap_matrix:
        return None

    # Extract method pairs and percentages
    pairs = list(overlap_matrix.keys())
    methods = set()
    for pair in pairs:
        m1, m2 = pair.split("_vs_")
        methods.add(m1)
        methods.add(m2)

    methods = sorted(list(methods))
    n = len(methods)

    # Create matrix
    matrix = [[100 if i == j else 0 for j in range(n)] for i in range(n)]

    for pair, data in overlap_matrix.items():
        m1, m2 = pair.split("_vs_")
        i = methods.index(m1)
        j = methods.index(m2)
        overlap_pct = data["overlap_percentage"]
        matrix[i][j] = overlap_pct
        matrix[j][i] = overlap_pct

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=methods,
            y=methods,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text:.1f}%",
            textfont={"size": 12},
            colorbar=dict(title="Overlap %"),
        )
    )

    fig.update_layout(
        title="Result Overlap Between Methods",
        xaxis_title="Method",
        yaxis_title="Method",
        height=500,
    )

    return fig


def display_chunks(chunks: List[Dict[str, Any]], method: str):
    """Display retrieved chunks with scores."""
    st.subheader(f"{method} Results")

    for i, chunk in enumerate(chunks, 1):
        score = chunk["score"]
        text = chunk["text"]
        chunk_id = chunk.get("id", "unknown")

        with st.expander(f"**Rank #{i}** | Score: {score:.4f} | ID: {chunk_id}"):
            st.markdown(f"```\n{text}\n```")

            # Display metadata if available
            if "metadata" in chunk and chunk["metadata"]:
                st.caption(f"Metadata: {chunk['metadata']}")


def main():
    """Main Streamlit application."""

    # Header
    st.title("🎯 Dartboard RAG - Retrieval Method Comparison")
    st.markdown(
        """
        Compare different retrieval methods side-by-side to evaluate their performance
        on your queries. Choose from BM25 (sparse), Dense (semantic), Hybrid (RRF fusion),
        and Dartboard (proprietary algorithm).
        """
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # API connection status
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            st.sidebar.success("✅ Connected to API")
            health_data = health_response.json()
            st.sidebar.metric(
                "Documents Indexed", health_data.get("vector_store_count", 0)
            )
        else:
            st.sidebar.error("❌ API Connection Failed")
    except:
        st.sidebar.error("❌ Cannot reach API")

    st.sidebar.divider()

    # Method selection
    st.sidebar.subheader("Retrieval Methods")
    available_methods = {
        "BM25": "bm25",
        "Dense": "dense",
        "Hybrid (RRF)": "hybrid",
        "Dartboard": "dartboard",
    }

    selected_methods = []
    for display_name, method_id in available_methods.items():
        if st.sidebar.checkbox(
            display_name, value=(method_id in ["bm25", "dense", "hybrid"])
        ):
            selected_methods.append(method_id)

    # Retrieval parameters
    st.sidebar.subheader("Parameters")
    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=5)
    use_reranker = st.sidebar.checkbox("Use Cross-Encoder Reranking", value=False)

    if use_reranker:
        st.sidebar.info("🔄 Cross-encoder will rerank results for all methods")

    # Main query input
    st.header("🔍 Query Input")
    query = st.text_area(
        "Enter your query:",
        placeholder="What is machine learning?",
        height=100,
    )

    # Search button
    if st.button("🚀 Compare Methods", type="primary", width="stretch"):
        if not query.strip():
            st.warning("⚠️ Please enter a query")
        elif not selected_methods:
            st.warning("⚠️ Please select at least one retrieval method")
        else:
            with st.spinner("Running comparison..."):
                # Call API
                response = call_compare_api(
                    query, selected_methods, top_k, use_reranker
                )

                if response:
                    # Store in session state
                    st.session_state["last_response"] = response
                    st.session_state["last_query"] = query

    # Display results if available
    if "last_response" in st.session_state:
        response = st.session_state["last_response"]
        query = st.session_state["last_query"]

        st.divider()
        st.header("📊 Comparison Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Query", f'"{query[:30]}..."' if len(query) > 30 else f'"{query}"'
            )
        with col2:
            st.metric(
                "Methods Compared", response["comparison_metrics"]["methods_compared"]
            )
        with col3:
            st.metric("Total Time", f"{response['total_time_ms']:.0f} ms")
        with col4:
            st.metric(
                "Avg Latency",
                f"{response['comparison_metrics']['avg_latency_ms']:.0f} ms",
            )

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📝 Results", "📈 Metrics", "🔗 Overlap", "⚡ Performance"]
        )

        with tab1:
            st.subheader("Retrieved Documents by Method")

            # Display results side-by-side
            results = response["results"]
            n_methods = len(results)

            if n_methods == 1:
                display_chunks(results[0]["chunks"], results[0]["method"])
            elif n_methods == 2:
                col1, col2 = st.columns(2)
                with col1:
                    display_chunks(results[0]["chunks"], results[0]["method"])
                with col2:
                    display_chunks(results[1]["chunks"], results[1]["method"])
            elif n_methods == 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    display_chunks(results[0]["chunks"], results[0]["method"])
                with col2:
                    display_chunks(results[1]["chunks"], results[1]["method"])
                with col3:
                    display_chunks(results[2]["chunks"], results[2]["method"])
            else:  # 4 methods
                col1, col2 = st.columns(2)
                with col1:
                    display_chunks(results[0]["chunks"], results[0]["method"])
                    if len(results) > 2:
                        display_chunks(results[2]["chunks"], results[2]["method"])
                with col2:
                    display_chunks(results[1]["chunks"], results[1]["method"])
                    if len(results) > 3:
                        display_chunks(results[3]["chunks"], results[3]["method"])

        with tab2:
            st.subheader("Score Distributions")
            fig_scores = create_score_distribution_chart(results)
            st.plotly_chart(fig_scores, width="stretch")

            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_data = []
            for result in results:
                metrics_data.append(
                    {
                        "Method": result["method"],
                        "Latency (ms)": f"{result['latency_ms']:.2f}",
                        "Avg Score": (
                            f"{sum(result['scores']) / len(result['scores']):.4f}"
                            if result["scores"]
                            else "N/A"
                        ),
                        "Max Score": (
                            f"{max(result['scores']):.4f}"
                            if result["scores"]
                            else "N/A"
                        ),
                        "Min Score": (
                            f"{min(result['scores']):.4f}"
                            if result["scores"]
                            else "N/A"
                        ),
                        "Results Count": len(result["chunks"]),
                    }
                )

            df = pd.DataFrame(metrics_data)
            st.dataframe(df, width="stretch", hide_index=True)

        with tab3:
            st.subheader("Result Overlap Analysis")

            overlap_matrix = response["comparison_metrics"].get("overlap", {})
            if overlap_matrix:
                fig_heatmap = create_overlap_heatmap(overlap_matrix)
                st.plotly_chart(fig_heatmap, width="stretch")

                # Detailed overlap table
                st.subheader("Overlap Details")
                overlap_data = []
                for pair, data in overlap_matrix.items():
                    overlap_data.append(
                        {
                            "Method Pair": pair.replace("_vs_", " ↔ "),
                            "Overlap Count": data["overlap_count"],
                            "Overlap %": f"{data['overlap_percentage']:.1f}%",
                        }
                    )

                df_overlap = pd.DataFrame(overlap_data)
                st.dataframe(df_overlap, width="stretch", hide_index=True)
            else:
                st.info("Overlap analysis requires at least 2 methods")

            # Unique chunks metric
            st.metric(
                "Total Unique Chunks",
                response["comparison_metrics"]["total_unique_chunks"],
            )

        with tab4:
            st.subheader("Performance Comparison")

            fig_latency = create_latency_chart(results)
            st.plotly_chart(fig_latency, width="stretch")

            # Performance insights
            st.subheader("Performance Insights")

            fastest = min(results, key=lambda x: x["latency_ms"])
            slowest = max(results, key=lambda x: x["latency_ms"])

            col1, col2 = st.columns(2)
            with col1:
                st.success(
                    f"⚡ **Fastest**: {fastest['method']} ({fastest['latency_ms']:.0f} ms)"
                )
            with col2:
                st.warning(
                    f"🐢 **Slowest**: {slowest['method']} ({slowest['latency_ms']:.0f} ms)"
                )

            if use_reranker:
                st.info(
                    "🔄 Cross-encoder reranking was applied to all methods, "
                    "which adds additional latency but improves precision."
                )

    # Footer
    st.divider()
    st.caption(
        "🎯 Dartboard RAG System | Compare retrieval methods to optimize your search performance"
    )


if __name__ == "__main__":
    main()
