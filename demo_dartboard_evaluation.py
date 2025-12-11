"""
Comprehensive Dartboard RAG evaluation demo.

Tests the complete system including:
- Synthetic dataset generation
- Dartboard retrieval
- Evaluation metrics
"""

import numpy as np
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.synthetic import SyntheticConfig, SyntheticDatasetGenerator
from dartboard.evaluation.metrics import RetrievalEvaluator, ComparisonEvaluator


def main():
    print("=" * 60)
    print("🎯 Dartboard RAG - Comprehensive Evaluation Demo")
    print("=" * 60)
    print()

    # Initialize embedding model
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")
    print()

    # Generate synthetic dataset
    print("🔨 Generating synthetic clustered dataset...")
    synthetic_config = SyntheticConfig(
        num_clusters=3, passages_per_cluster=5, cluster_variance=0.2, seed=42
    )
    generator = SyntheticDatasetGenerator(synthetic_config, embedding_model)
    dataset = generator.generate_clustered_dataset()
    print(
        f"✓ Generated dataset: {len(dataset.chunks)} chunks, {len(dataset.queries)} queries"
    )
    print(f"   Clusters: {synthetic_config.num_clusters}")
    print(f"   Passages per cluster: {synthetic_config.passages_per_cluster}")
    print()

    # Test different sigma values
    print("🔬 Testing Dartboard with different σ values...")
    sigma_values = [0.5, 1.0, 2.0]
    system_results = {}

    for sigma in sigma_values:
        print(f"\n  Testing σ = {sigma}...")
        config = DartboardConfig(
            sigma=sigma, top_k=3, triage_k=10, reranker_type="cosine"
        )
        retriever = DartboardRetriever(config, embedding_model)

        # Retrieve for each query
        results = []
        for query in dataset.queries:
            result = retriever.retrieve(query, dataset.chunks)
            results.append(result)

        system_results[f"Dartboard_σ={sigma}"] = results
        print(f"  ✓ Retrieved {len(results)} queries with top_k={config.top_k}")

    print()

    # Evaluate all systems
    print("📊 Evaluating retrieval performance...")
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    comparison = ComparisonEvaluator()

    reports = comparison.compare_systems(system_results, dataset)

    # Display results
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    for system_name, report in reports.items():
        print(f"\n{system_name}:")
        print("-" * 40)
        print(f"  NDCG:       {report.metrics['ndcg']:.4f}")
        print(f"  MAP:        {report.metrics['map']:.4f}")
        print(f"  P@1:        {report.metrics['precision@1']:.4f}")
        print(f"  P@3:        {report.metrics['precision@3']:.4f}")
        print(f"  R@3:        {report.metrics['recall@3']:.4f}")
        print(f"  Diversity:  {report.metrics['diversity']:.4f}")

    # Comparison table
    print()
    print("=" * 60)
    print(comparison.generate_comparison_table(reports))
    print("=" * 60)
    print()

    # Detailed example for one query
    print("📝 Detailed Example - Query 0:")
    print("-" * 40)
    query_idx = 0
    query = dataset.queries[query_idx]
    print(f"Query: {query}")
    print()

    # Show results for different sigma values
    for sigma in sigma_values:
        system_name = f"Dartboard_σ={sigma}"
        result = system_results[system_name][query_idx]
        print(f"\n{system_name} Results:")
        for i, (chunk, score) in enumerate(zip(result.chunks, result.scores), 1):
            cluster = chunk.metadata.get("cluster", "?")
            print(f"  {i}. [Score: {score:.4f}, Cluster: {cluster}]")
            print(f"     {chunk.text[:80]}...")

    print()
    print("=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
