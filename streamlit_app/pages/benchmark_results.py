"""
Streamlit Page for Viewing Benchmark Evaluation Results.

This page displays benchmark results from JSON files, showing comparative
metrics, charts, and detailed analysis of retrieval method performance.
"""

import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def get_metric_explanation(metric_name: str) -> str:
    """
    Return detailed explanation for a given metric.

    Args:
        metric_name: Name of the metric (e.g., "MRR@100", "NDCG@10", "ILD")

    Returns:
        Markdown-formatted explanation string
    """
    # Extract base metric name (remove @K suffix)
    base_metric = metric_name.split("@")[0]

    explanations = {
        "MRR": """
**Mean Reciprocal Rank (MRR@K)**

Measures how quickly a relevant document appears in the ranked results.

**Formula:** MRR = 1 / rank of first relevant document

**Range:** 0.0 to 1.0 (higher is better)

**Example:**
- If the first relevant doc is at position 1: MRR = 1.0
- If the first relevant doc is at position 2: MRR = 0.5
- If the first relevant doc is at position 10: MRR = 0.1

**Use Case:** Important for question answering and search where users typically look at only the top few results.
""",
        "MAP": """
**Mean Average Precision (MAP@K)**

Measures both precision and the ranking quality of relevant documents.

**Formula:** Average of precision values at positions where relevant documents appear

**Range:** 0.0 to 1.0 (higher is better)

**Example:** If relevant docs appear at positions 1, 3, and 5:
- Precision@1 = 1/1 = 1.0
- Precision@3 = 2/3 = 0.667
- Precision@5 = 3/5 = 0.6
- MAP = (1.0 + 0.667 + 0.6) / 3 = 0.756

**Use Case:** Excellent for evaluating systems where multiple relevant documents exist and their ranking matters.
""",
        "NDCG": """
**Normalized Discounted Cumulative Gain (NDCG@K)**

Measures ranking quality with position-based discounting - relevant docs ranked higher are more valuable.

**Formula:** DCG / Ideal DCG, where DCG = Σ (relevance / log₂(position + 1))

**Range:** 0.0 to 1.0 (higher is better)

**Key Insight:** Documents appearing earlier in the ranking contribute more to the score due to logarithmic discounting.

**Example:**
- Relevant doc at position 1: gain = 1.0 / log₂(2) = 1.0
- Relevant doc at position 2: gain = 1.0 / log₂(3) = 0.631
- Relevant doc at position 10: gain = 1.0 / log₂(11) = 0.289

**Use Case:** Industry standard for evaluating search engines and recommender systems.
""",
        "Recall": """
**Recall@K**

Measures what fraction of all relevant documents were retrieved in the top-K results.

**Formula:** Recall@K = (# relevant docs in top-K) / (total # relevant docs)

**Range:** 0.0 to 1.0 (higher is better)

**Example:** If there are 10 relevant docs total and 7 appear in top-K:
- Recall@K = 7/10 = 0.7

**Trade-off:** Higher K typically increases recall but may decrease precision.

**Use Case:** Important when you need to ensure all relevant information is retrieved (e.g., medical literature search, legal discovery).
""",
        "Precision": """
**Precision@K**

Measures what fraction of the top-K retrieved documents are actually relevant.

**Formula:** Precision@K = (# relevant docs in top-K) / K

**Range:** 0.0 to 1.0 (higher is better)

**Example:** If top-10 results contain 7 relevant documents:
- Precision@10 = 7/10 = 0.7

**Trade-off:** Precision and recall have an inverse relationship - improving one often decreases the other.

**Use Case:** Critical when showing irrelevant results is costly (e.g., spam detection, high-stakes decision making).
""",
        "ILD": """
**Intra-List Diversity (ILD)**

Measures how diverse (dissimilar) the retrieved documents are from each other.

**Formula:** Average pairwise cosine distance between all document embeddings in results

**Range:** 0.0 to 1.0 (higher = more diverse)

**Calculation:** ILD = (1 / (|R| × (|R|-1))) × Σ dissimilarity(docᵢ, docⱼ)

**Why It Matters:** High relevance but low diversity means redundant results - different docs saying the same thing.

**Example:**
- All documents about the same subtopic: ILD ≈ 0.2
- Documents covering different aspects: ILD ≈ 0.6-0.8

**Use Case:** Important for exploratory search, news aggregation, and recommendation systems where variety matters.
""",
        "Alpha-NDCG": """
**Alpha-NDCG@10**

Balances relevance and diversity by penalizing redundant documents similar to already-seen results.

**Formula:** Modified NDCG where gain = relevance × (α + (1-α) × novelty)

**Parameters:**
- α = 0.5: Equal weight to relevance and diversity
- α = 0.0: Pure diversity (novelty only)
- α = 1.0: Pure relevance (standard NDCG)

**Range:** 0.0 to 1.0 (higher is better)

**Novelty Calculation:** 1 - max_similarity to previously seen documents

**Use Case:** Ideal for comparing retrieval systems on their ability to provide both relevant AND diverse results. Used in research on diversity-aware ranking.
""",
        "AP": """
**Average Precision (AP)**

Similar to MAP but for a single query - measures precision at each relevant document position.

**Formula:** (1 / # relevant docs) × Σ (Precision@i × is_relevant_i)

**Range:** 0.0 to 1.0 (higher is better)

**Use Case:** Building block for calculating MAP across multiple queries.
""",
    }

    return explanations.get(base_metric, f"No explanation available for {metric_name}")


def render_metric_with_help(metric_name: str, col_width: str = "90%"):
    """
    Render a metric name with a help popover icon.

    Args:
        metric_name: Name of the metric
        col_width: Width of the metric name column (remaining space for popover)
    """
    col1, col2 = st.columns([col_width, f"{100 - int(col_width.rstrip('%'))}%"])
    with col1:
        st.write(metric_name)
    with col2:
        with st.popover("ℹ️"):
            st.markdown(get_metric_explanation(metric_name))


# Page configuration
st.set_page_config(
    page_title="Benchmark Results - Dartboard RAG",
    page_icon="📊",
    layout="wide",
)


def load_benchmark_results(results_dir: str = "benchmarks/results") -> List[Dict]:
    """Load all benchmark result JSON files."""
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    results = []
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["filename"] = json_file.name
                results.append(data)
        except Exception as e:
            st.warning(f"Error loading {json_file.name}: {e}")

    # Sort by timestamp descending (newest first)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return results


def create_metrics_comparison_chart(
    report: Dict[str, Any], selected_metrics: List[str]
) -> go.Figure:
    """Create grouped bar chart comparing metrics across methods."""
    methods = report["methods"]
    all_metrics = report["metrics"]

    fig = go.Figure()

    for method in methods:
        method_data = all_metrics[method]
        if "error" in method_data:
            continue

        values = [method_data.get(m, 0.0) for m in selected_metrics]
        fig.add_trace(
            go.Bar(
                name=method.upper(),
                x=selected_metrics,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title=f"Retrieval Method Comparison - {report['dataset']}",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_ndcg_comparison(report: Dict[str, Any]) -> go.Figure:
    """Create NDCG@K line chart."""
    fig = go.Figure()

    methods = report["methods"]
    k_values = report.get("k_values", [1, 5, 10, 20, 100])

    for method in methods:
        method_data = report["metrics"][method]
        if "error" in method_data:
            continue

        ndcg_values = []
        for k in k_values:
            ndcg = method_data.get(f"NDCG@{k}", 0.0)
            ndcg_values.append(ndcg)

        fig.add_trace(
            go.Scatter(
                x=[f"@{k}" for k in k_values],
                y=ndcg_values,
                mode="lines+markers",
                name=method.upper(),
                marker=dict(size=10),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=f"NDCG@K Comparison - {report['dataset']}",
        xaxis_title="K",
        yaxis_title="NDCG Score",
        height=400,
        showlegend=True,
    )

    return fig


def create_recall_precision_curve(report: Dict[str, Any]) -> go.Figure:
    """Create recall-precision curves."""
    fig = go.Figure()

    methods = report["methods"]
    k_values = report.get("k_values", [1, 5, 10, 20, 100])

    for method in methods:
        method_data = report["metrics"][method]
        if "error" in method_data:
            continue

        recalls = []
        precisions = []

        for k in k_values:
            recall = method_data.get(f"Recall@{k}", 0.0)
            precision = method_data.get(f"Precision@{k}", 0.0)
            recalls.append(recall)
            precisions.append(precision)

        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines+markers",
                name=method.upper(),
                marker=dict(size=10),
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=f"Recall-Precision Curves - {report['dataset']}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400,
        showlegend=True,
    )

    return fig


def create_summary_table(report: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table with key metrics."""
    methods = report["methods"]
    data = []

    key_metrics = ["MRR@100", "MAP@100", "NDCG@10", "Recall@10", "Precision@10"]

    for method in methods:
        method_data = report["metrics"][method]
        if "error" in method_data:
            continue

        row = {"Method": method.upper()}
        for metric in key_metrics:
            row[metric] = method_data.get(metric, 0.0)

        data.append(row)

    return pd.DataFrame(data)


def main():
    """Main Streamlit application."""

    # Header
    st.title("📊 Benchmark Evaluation Results")
    st.markdown(
        """
        View and analyze benchmark evaluation results comparing different retrieval methods
        on standard IR datasets (MS MARCO, BEIR).
        """
    )

    # Load results
    results = load_benchmark_results()

    if not results:
        st.warning(
            "⚠️ No benchmark results found. Run benchmarks using:\n\n"
            "```bash\n"
            "python benchmarks/scripts/run_benchmark.py --dataset msmarco --sample 100\n"
            "```"
        )
        return

    st.success(f"✅ Found {len(results)} benchmark result(s)")

    # Sidebar - result selection
    st.sidebar.header("📁 Select Benchmark")

    result_options = {}
    for i, result in enumerate(results):
        label = f"{result['dataset']} - {result.get('timestamp', 'Unknown')} ({result['num_queries']} queries)"
        result_options[label] = i

    selected_label = st.sidebar.selectbox(
        "Choose a benchmark result:",
        options=list(result_options.keys()),
    )

    selected_idx = result_options[selected_label]
    report = results[selected_idx]

    # Display report metadata
    st.sidebar.divider()
    st.sidebar.subheader("📋 Benchmark Info")
    st.sidebar.metric("Dataset", report["dataset"])
    st.sidebar.metric("Queries", report["num_queries"])
    st.sidebar.metric("Documents", report["num_documents"])
    st.sidebar.metric("Methods", len(report["methods"]))
    st.sidebar.caption(f"Timestamp: {report.get('timestamp', 'N/A')}")

    # Main content
    st.divider()
    st.header(f"Results: {report['dataset']}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", report["dataset"])
    with col2:
        st.metric("Queries", f"{report['num_queries']:,}")
    with col3:
        st.metric("Documents", f"{report['num_documents']:,}")
    with col4:
        st.metric("Methods", len(report["methods"]))

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Summary", "📈 Charts", "📉 Detailed Metrics", "🔍 Raw Data"]
    )

    with tab1:
        st.subheader("Summary Metrics")

        # Add explanation expander
        with st.expander("📖 Learn About Metrics"):
            st.markdown(
                """
            Click the **ℹ️ icons** next to each metric name below to learn:
            - What the metric measures
            - How it's calculated
            - When to use it
            - Example interpretations

            **Quick Overview:**
            - **MRR**: How quickly you find the first relevant result
            - **MAP**: Quality of ranking across all relevant results
            - **NDCG**: Ranking quality with position discounting
            - **Recall**: What % of relevant docs were found
            - **Precision**: What % of results are relevant
            - **ILD**: How diverse (non-redundant) results are
            - **Alpha-NDCG**: Balance of relevance AND diversity
            """
            )

        # Create summary table
        df = create_summary_table(report)

        # Display metric headers with help icons
        st.markdown("##### Metric Definitions")
        metric_cols = [col for col in df.columns if col != "Method"]
        cols = st.columns(len(metric_cols))

        for i, metric in enumerate(metric_cols):
            with cols[i]:
                with st.popover(f"**{metric}** ℹ️"):
                    st.markdown(get_metric_explanation(metric))

        st.divider()

        # Style the dataframe
        st.dataframe(
            df.style.format({col: "{:.4f}" for col in df.columns if col != "Method"}),
            width="stretch",
            hide_index=True,
        )

        # Highlight best performers
        st.subheader("🏆 Best Performers")

        key_metrics = ["MRR@100", "NDCG@10", "Recall@10"]
        cols = st.columns(len(key_metrics))

        for i, metric in enumerate(key_metrics):
            with cols[i]:
                best_method = None
                best_value = 0

                for method in report["methods"]:
                    method_data = report["metrics"][method]
                    if "error" not in method_data:
                        value = method_data.get(metric, 0)
                        if value > best_value:
                            best_value = value
                            best_method = method

                if best_method:
                    # Display metric with popover
                    metric_col, popover_col = st.columns([0.85, 0.15])
                    with metric_col:
                        st.metric(
                            metric,
                            f"{best_method.upper()}",
                            f"{best_value:.4f}",
                        )
                    with popover_col:
                        with st.popover("ℹ️"):
                            st.markdown(get_metric_explanation(metric))

    with tab2:
        st.subheader("Performance Charts")

        # Metric selector
        all_metric_names = set()
        for method_metrics in report["metrics"].values():
            if "error" not in method_metrics:
                all_metric_names.update(method_metrics.keys())

        available_metrics = sorted(list(all_metric_names))

        selected_metrics = st.multiselect(
            "Select metrics to compare:",
            options=available_metrics,
            default=["MRR@100", "MAP@100", "NDCG@10", "Recall@10", "Precision@10"],
        )

        if selected_metrics:
            fig_comparison = create_metrics_comparison_chart(report, selected_metrics)
            st.plotly_chart(fig_comparison, width="stretch")

        # NDCG@K chart
        st.subheader("NDCG@K Progression")
        fig_ndcg = create_ndcg_comparison(report)
        st.plotly_chart(fig_ndcg, width="stretch")

        # Recall-Precision curve
        st.subheader("Recall-Precision Trade-off")
        fig_rp = create_recall_precision_curve(report)
        st.plotly_chart(fig_rp, width="stretch")

    with tab3:
        st.subheader("Detailed Metrics by Method")

        for method in report["methods"]:
            method_data = report["metrics"][method]

            with st.expander(f"**{method.upper()}**"):
                if "error" in method_data:
                    st.error(f"Error: {method_data['error']}")
                else:
                    # Convert to DataFrame for nice display
                    metrics_df = pd.DataFrame(
                        [
                            {"Metric": k, "Value": v}
                            for k, v in sorted(method_data.items())
                        ]
                    )

                    st.dataframe(
                        metrics_df.style.format({"Value": "{:.4f}"}),
                        width="stretch",
                        hide_index=True,
                    )

    with tab4:
        st.subheader("Raw Benchmark Data")
        st.json(report)

    # Footer
    st.divider()
    st.caption("🎯 Dartboard RAG - Benchmark Evaluation System")


if __name__ == "__main__":
    main()
