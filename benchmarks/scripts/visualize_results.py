"""
Visualization Generator for Benchmark Results.

Creates charts and reports from benchmark evaluation results.

Usage:
    python benchmarks/scripts/visualize_results.py --results benchmarks/results/msmarco_20241125.json
    python benchmarks/scripts/visualize_results.py --results benchmarks/results/*.json --output report.html
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Creates visualizations from benchmark results."""

    def __init__(self, results_dir: str = "benchmarks/results"):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)

    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(filepath, "r") as f:
            return json.load(f)

    def create_metrics_comparison_chart(
        self, report: Dict[str, Any], metrics: List[str] = None
    ) -> go.Figure:
        """
        Create grouped bar chart comparing metrics across methods.

        Args:
            report: Benchmark report
            metrics: Specific metrics to plot (default: all)

        Returns:
            Plotly figure
        """
        methods = report["methods"]
        all_metrics = report["metrics"]

        # Get metric names
        if metrics is None:
            metrics = []
            for method_metrics in all_metrics.values():
                if "error" not in method_metrics:
                    metrics.extend(method_metrics.keys())
            metrics = sorted(set(metrics))

        # Create figure
        fig = go.Figure()

        for method in methods:
            method_data = all_metrics[method]
            if "error" in method_data:
                continue

            values = [method_data.get(m, 0.0) for m in metrics]
            fig.add_trace(
                go.Bar(
                    name=method.upper(),
                    x=metrics,
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
            height=600,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def create_recall_precision_curve(self, report: Dict[str, Any]) -> go.Figure:
        """
        Create recall-precision curves for each method.

        Args:
            report: Benchmark report

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        methods = report["methods"]
        k_values = report.get("k_values", [1, 5, 10, 20, 100])

        for method in methods:
            method_data = report["metrics"][method]
            if "error" in method_data:
                continue

            # Extract recall and precision at each k
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
            height=500,
            showlegend=True,
        )

        return fig

    def create_ndcg_comparison(self, report: Dict[str, Any]) -> go.Figure:
        """
        Create NDCG@K comparison chart.

        Args:
            report: Benchmark report

        Returns:
            Plotly figure
        """
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
            height=500,
            showlegend=True,
        )

        return fig

    def create_summary_table(self, report: Dict[str, Any]) -> pd.DataFrame:
        """
        Create summary table with key metrics.

        Args:
            report: Benchmark report

        Returns:
            Pandas DataFrame
        """
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

    def generate_html_report(
        self,
        reports: List[Dict[str, Any]],
        output_file: str = "benchmarks/results/report.html",
    ):
        """
        Generate comprehensive HTML report.

        Args:
            reports: List of benchmark reports
            output_file: Output HTML file path
        """
        logger.info(f"Generating HTML report for {len(reports)} benchmark(s)...")

        html_parts = []

        # Header
        timestamp = reports[0]["timestamp"] if reports else "N/A"
        html_parts.append(
            f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dartboard RAG - Benchmark Results</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .dataset-section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .chart {{
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <h1>🎯 Dartboard RAG - Benchmark Evaluation Results</h1>
    <p><strong>Generated:</strong> {timestamp}</p>
"""
        )

        # Process each report
        for i, report in enumerate(reports):
            html_parts.append(
                f"""
    <div class="dataset-section">
        <h2>Dataset: {report['dataset']}</h2>
        <p><strong>Queries:</strong> {report['num_queries']} | <strong>Documents:</strong> {report['num_documents']}</p>
"""
            )

            # Summary table
            df = self.create_summary_table(report)
            html_parts.append("<h3>Summary Metrics</h3>")
            html_parts.append(
                df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
            )

            # Charts
            metrics_chart = self.create_metrics_comparison_chart(report)
            html_parts.append(f'<div class="chart" id="metrics_{i}"></div>')
            html_parts.append(
                f'<script>Plotly.newPlot("metrics_{i}", {metrics_chart.to_json()});</script>'
            )

            ndcg_chart = self.create_ndcg_comparison(report)
            html_parts.append(f'<div class="chart" id="ndcg_{i}"></div>')
            html_parts.append(
                f'<script>Plotly.newPlot("ndcg_{i}", {ndcg_chart.to_json()});</script>'
            )

            rp_curve = self.create_recall_precision_curve(report)
            html_parts.append(f'<div class="chart" id="rp_{i}"></div>')
            html_parts.append(
                f'<script>Plotly.newPlot("rp_{i}", {rp_curve.to_json()});</script>'
            )

            html_parts.append("    </div>")

        # Footer
        html_parts.append(
            """
</body>
</html>
"""
        )

        # Write file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(html_parts))

        logger.info(f"✅ HTML report saved to: {output_path}")
        logger.info(f"📊 Open in browser: file://{output_path.absolute()}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")

    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Path(s) to benchmark result JSON files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/report.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    visualizer = BenchmarkVisualizer()

    # Load all results
    reports = []
    for result_file in args.results:
        logger.info(f"Loading: {result_file}")
        report = visualizer.load_results(result_file)
        reports.append(report)

    # Generate report
    visualizer.generate_html_report(reports, output_file=args.output)


if __name__ == "__main__":
    main()
