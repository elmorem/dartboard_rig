"""
Scalability stress test for Dartboard RAG.

Tests performance with larger corpus sizes to validate
the triage mechanism and overall system efficiency.
"""

import time
import numpy as np
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.synthetic import SyntheticConfig, SyntheticDatasetGenerator


def main():
    print("=" * 70)
    print("⚡ Dartboard Scalability Stress Test")
    print("=" * 70)
    print()

    # Initialize embedding model
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")
    print()

    # Test different corpus sizes
    corpus_sizes = [50, 100, 200, 500]

    results = []

    for corpus_size in corpus_sizes:
        print("=" * 70)
        print(f"Testing with corpus size: {corpus_size}")
        print("=" * 70)
        print()

        # Generate dataset
        print(f"🔨 Generating {corpus_size} passages...")
        num_clusters = max(5, corpus_size // 20)
        passages_per_cluster = corpus_size // num_clusters

        synthetic_config = SyntheticConfig(
            num_clusters=num_clusters,
            passages_per_cluster=passages_per_cluster,
            embedding_dim=get_embedding_config().embedding_dim,
            seed=42,
        )
        generator = SyntheticDatasetGenerator(synthetic_config)
        dataset = generator.generate_clustered_dataset()

        actual_size = len(dataset.chunks)
        print(f"✓ Generated {actual_size} passages in {num_clusters} clusters")
        print()

        # Configure retriever with triage
        triage_k = min(100, actual_size)
        config = DartboardConfig(
            sigma=1.0, top_k=10, triage_k=triage_k, reranker_type="cosine"
        )
        retriever = DartboardRetriever(config, embedding_model)

        print(f"🎯 Retrieval configuration:")
        print(f"   top_k: {config.top_k}")
        print(f"   triage_k: {config.triage_k}")
        print()

        # Run retrieval and measure time
        query = "Test query for performance measurement"

        print("⏱️  Running retrieval...")
        start_time = time.time()
        result = retriever.retrieve(query, dataset.chunks)
        elapsed_time = time.time() - start_time

        print(f"✓ Retrieval completed in {elapsed_time:.4f}s")
        print(f"   Retrieved {len(result.chunks)} passages")
        print(f"   Throughput: {actual_size / elapsed_time:.1f} passages/sec")
        print()

        # Store results
        results.append(
            {
                "corpus_size": actual_size,
                "time": elapsed_time,
                "throughput": actual_size / elapsed_time,
            }
        )

    # Summary
    print("=" * 70)
    print("📊 Performance Summary")
    print("=" * 70)
    print()
    print(f"{'Corpus Size':<15} {'Time (s)':<12} {'Throughput (p/s)':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r['corpus_size']:<15} {r['time']:<12.4f} {r['throughput']:<20.1f}")

    print()
    print("=" * 70)
    print("✓ Scalability test complete!")
    print("=" * 70)
    print()
    print("Note: Triage mechanism (KNN) provides O(n log k) performance")
    print("      for large corpora, making Dartboard efficient even at scale.")


if __name__ == "__main__":
    main()
