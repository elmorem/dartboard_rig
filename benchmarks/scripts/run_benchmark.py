"""
Benchmark Evaluation Runner for Retrieval Methods.

This script runs comprehensive benchmarks comparing BM25, Dense, Hybrid,
and Dartboard retrieval methods on standard IR datasets (MS MARCO, BEIR).

Usage:
    python benchmarks/scripts/run_benchmark.py --dataset msmarco --sample 1000
    python benchmarks/scripts/run_benchmark.py --dataset beir-scifact --methods bm25 dense dartboard
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.retrieval.base import Chunk
from dartboard.evaluation.metrics import evaluate_batch
from dartboard.evaluation.datasets import MSMARCOLoader, BEIRLoader
from dartboard.storage.vector_store import FAISSStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks on retrieval methods."""

    def __init__(
        self,
        data_dir: str = "benchmarks/data",
        results_dir: str = "benchmarks/results",
    ):
        """
        Initialize benchmark runner.

        Args:
            data_dir: Directory for cached datasets
            results_dir: Directory for benchmark results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(
        self, dataset_name: str, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load evaluation dataset.

        Args:
            dataset_name: Dataset identifier (e.g., 'msmarco', 'beir-scifact')
            sample_size: Optional sample size for faster testing

        Returns:
            Dict with queries, documents, and qrels
        """
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name == "msmarco":
            loader = MSMARCOLoader(data_dir=str(self.data_dir / "msmarco"))
            dataset = loader.load_dev_small()
        elif dataset_name.startswith("beir-"):
            beir_dataset = dataset_name.replace("beir-", "")
            loader = BEIRLoader(data_dir=str(self.data_dir / "beir"))
            dataset = loader.load_dataset(beir_dataset, split="test")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Sample if requested - only sample queries that have relevance judgments
        if sample_size and sample_size < len(dataset.queries):
            logger.info(f"Sampling {sample_size} queries from {len(dataset.queries)}")

            # Get queries that have qrels
            queries_with_qrels = {qrel.query_id for qrel in dataset.qrels}
            filtered_queries = [
                q for q in dataset.queries if q.id in queries_with_qrels
            ]

            if len(filtered_queries) < sample_size:
                logger.warning(
                    f"Only {len(filtered_queries)} queries have relevance judgments, "
                    f"sampling all of them instead of {sample_size}"
                )
                dataset.queries = filtered_queries
            else:
                dataset.queries = filtered_queries[:sample_size]

            # Filter qrels to sampled queries
            sampled_query_ids = {q.id for q in dataset.queries}
            dataset.qrels = [
                qrel for qrel in dataset.qrels if qrel.query_id in sampled_query_ids
            ]

        logger.info(
            f"Dataset loaded: {len(dataset.queries)} queries, "
            f"{len(dataset.documents)} documents"
        )

        return dataset

    def prepare_corpus(self, dataset) -> List[Chunk]:
        """
        Convert dataset documents to Chunk objects.

        Args:
            dataset: Evaluation dataset

        Returns:
            List of Chunk objects
        """
        logger.info("Preparing corpus...")
        chunks = []

        for doc in dataset.documents:
            chunk = Chunk(
                id=doc.id, text=doc.text, metadata={"title": doc.title, **doc.metadata}
            )
            chunks.append(chunk)

        logger.info(f"Prepared {len(chunks)} chunks")
        return chunks

    def run_method(
        self, method_name: str, queries: List, chunks: List[Chunk], k: int = 10
    ) -> List[List[str]]:
        """
        Run a single retrieval method on all queries.

        Args:
            method_name: Method identifier
            queries: List of query objects
            chunks: Corpus chunks
            k: Number of results to retrieve

        Returns:
            List of result lists (doc IDs for each query)
        """
        logger.info(f"Running {method_name} retrieval...")

        # Initialize retriever
        if method_name == "bm25":
            retriever = BM25Retriever()
            retriever.fit(chunks)
        elif method_name == "dense":
            # Create dense retriever and generate embeddings
            dense = DenseRetriever()
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")

            # Generate embeddings for all chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = dense.encode_batch(texts, batch_size=32)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # Create FAISS vector store
            logger.info("Building FAISS index...")
            vector_store = FAISSStore(embedding_dim=embeddings.shape[1])
            vector_store.add(chunks)

            # Create retriever with vector store
            retriever = DenseRetriever(vector_store=vector_store)
        elif method_name == "hybrid":
            # BM25 component
            bm25 = BM25Retriever()
            bm25.fit(chunks)

            # Dense component
            dense_model = DenseRetriever()
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")

            # Generate embeddings for all chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = dense_model.encode_batch(texts, batch_size=32)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # Create FAISS vector store
            logger.info("Building FAISS index...")
            vector_store = FAISSStore(embedding_dim=embeddings.shape[1])
            vector_store.add(chunks)

            # Create dense retriever with vector store
            dense = DenseRetriever(vector_store=vector_store)

            # Create hybrid retriever
            retriever = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Run retrieval for each query
        all_results = []
        start_time = time.time()

        for i, query in enumerate(queries):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                qps = (i + 1) / elapsed
                logger.info(f"Processed {i + 1}/{len(queries)} queries ({qps:.1f} q/s)")

            result = retriever.retrieve(query.text, k=k)
            doc_ids = [chunk.id for chunk in result.chunks]
            all_results.append(doc_ids)

        total_time = time.time() - start_time
        avg_latency = (total_time / len(queries)) * 1000  # ms

        logger.info(
            f"{method_name} completed: {len(queries)} queries in {total_time:.1f}s "
            f"(avg: {avg_latency:.1f}ms/query)"
        )

        return all_results

    def evaluate_results(
        self,
        method_name: str,
        results: List[List[str]],
        dataset,
        k_values: List[int] = [1, 5, 10, 20, 100],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results using standard metrics.

        Args:
            method_name: Method identifier
            results: Retrieved doc IDs for each query
            dataset: Dataset with ground truth
            k_values: K values for metrics

        Returns:
            Dict of metric name -> score
        """
        logger.info(f"Evaluating {method_name} results...")

        # Prepare ground truth
        all_relevant_docs = []
        for query in dataset.queries:
            relevant = dataset.get_relevant_docs(query.id, min_relevance=1)
            all_relevant_docs.append(relevant)

        # Compute metrics
        metrics = evaluate_batch(results, all_relevant_docs, k_values=k_values)

        logger.info(f"{method_name} evaluation complete")
        for metric, value in sorted(metrics.items()):
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def run_benchmark(
        self,
        dataset_name: str,
        methods: List[str],
        sample_size: Optional[int] = None,
        k: int = 10,
        k_values: List[int] = [1, 5, 10, 20, 100],
    ) -> Dict[str, Any]:
        """
        Run complete benchmark comparing multiple methods.

        Args:
            dataset_name: Dataset to evaluate on
            methods: List of method names to compare
            sample_size: Optional sample size for testing
            k: Number of results to retrieve
            k_values: K values for evaluation metrics

        Returns:
            Complete benchmark results
        """
        logger.info("=" * 80)
        logger.info(f"Starting benchmark: {dataset_name}")
        logger.info(f"Methods: {methods}")
        logger.info("=" * 80)

        # Load dataset
        dataset = self.load_dataset(dataset_name, sample_size=sample_size)
        chunks = self.prepare_corpus(dataset)

        # Run each method
        all_results = {}
        all_metrics = {}

        for method in methods:
            try:
                # Retrieve
                results = self.run_method(method, dataset.queries, chunks, k=k)

                # Evaluate
                metrics = self.evaluate_results(
                    method, results, dataset, k_values=k_values
                )

                all_results[method] = results
                all_metrics[method] = metrics

            except Exception as e:
                logger.error(f"Error running {method}: {e}", exc_info=True)
                all_metrics[method] = {"error": str(e)}

        # Compile benchmark report
        report = {
            "dataset": dataset_name,
            "sample_size": sample_size or len(dataset.queries),
            "num_queries": len(dataset.queries),
            "num_documents": len(dataset.documents),
            "methods": methods,
            "k": k,
            "k_values": k_values,
            "metrics": all_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        self.save_results(report, dataset_name)

        logger.info("=" * 80)
        logger.info("Benchmark complete!")
        logger.info("=" * 80)

        return report

    def save_results(self, report: Dict[str, Any], dataset_name: str):
        """
        Save benchmark results to JSON file.

        Args:
            report: Benchmark report
            dataset_name: Dataset identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Results saved to: {filepath}")

    def print_comparison_table(self, report: Dict[str, Any]):
        """
        Print formatted comparison table.

        Args:
            report: Benchmark report
        """
        print("\n" + "=" * 80)
        print(f"BENCHMARK RESULTS: {report['dataset']}")
        print("=" * 80)
        print(
            f"Queries: {report['num_queries']} | Documents: {report['num_documents']}"
        )
        print("-" * 80)

        # Get all metrics
        methods = report["methods"]
        all_metric_names = set()
        for metrics in report["metrics"].values():
            if "error" not in metrics:
                all_metric_names.update(metrics.keys())

        # Print header
        header = f"{'Metric':<20}"
        for method in methods:
            header += f"{method:>15}"
        print(header)
        print("-" * 80)

        # Print metrics
        for metric_name in sorted(all_metric_names):
            row = f"{metric_name:<20}"
            for method in methods:
                metrics = report["metrics"][method]
                if "error" in metrics:
                    row += f"{'ERROR':>15}"
                else:
                    value = metrics.get(metric_name, 0.0)
                    row += f"{value:>15.4f}"
            print(row)

        print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run retrieval method benchmarks on standard datasets"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["msmarco", "beir-scifact", "beir-nfcorpus", "beir-fiqa"],
        help="Dataset to evaluate on",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "dense", "hybrid"],
        choices=["bm25", "dense", "hybrid", "dartboard"],
        help="Retrieval methods to compare",
    )

    parser.add_argument(
        "--sample", type=int, default=None, help="Sample N queries for faster testing"
    )

    parser.add_argument(
        "--k", type=int, default=10, help="Number of results to retrieve per query"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/data",
        help="Directory for cached datasets",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for benchmark results",
    )

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(data_dir=args.data_dir, results_dir=args.results_dir)

    report = runner.run_benchmark(
        dataset_name=args.dataset,
        methods=args.methods,
        sample_size=args.sample,
        k=args.k,
    )

    # Print results
    runner.print_comparison_table(report)


if __name__ == "__main__":
    main()
