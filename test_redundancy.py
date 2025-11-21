"""
Test Dartboard's ability to handle redundant/duplicate passages.

This tests whether Dartboard naturally selects diverse passages
even when the corpus contains many near-duplicates.
"""

import numpy as np
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.synthetic import SyntheticConfig, SyntheticDatasetGenerator
from dartboard.evaluation.metrics import DiversityAnalyzer


def main():
    print("=" * 70)
    print("🔬 Dartboard Redundancy Test")
    print("=" * 70)
    print()
    print("Testing whether Dartboard naturally deduplicates redundant passages...")
    print()

    # Initialize embedding model
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")
    print()

    # Generate redundant dataset
    print("🔨 Generating redundant dataset...")
    print("   - 10 unique passages")
    print("   - 5 near-duplicates per passage (50 total chunks)")
    print()

    synthetic_config = SyntheticConfig(
        num_clusters=2, passages_per_cluster=5, embedding_dim=384, seed=42
    )
    generator = SyntheticDatasetGenerator(synthetic_config)
    dataset = generator.generate_redundant_dataset(
        name="redundancy_test", redundancy_factor=5
    )

    print(f"✓ Generated dataset with {len(dataset.chunks)} chunks")
    print(f"   (10 unique × 5 copies each)")
    print()

    # Test with different sigma values
    print("🎯 Testing retrieval with different σ values...")
    print()

    sigma_values = [0.5, 1.0, 2.0, 5.0]
    query = "Query for passage 0"

    diversity_analyzer = DiversityAnalyzer()

    for sigma in sigma_values:
        print(f"σ = {sigma}:")
        print("-" * 50)

        config = DartboardConfig(
            sigma=sigma,
            top_k=10,  # Retrieve 10 passages
            triage_k=50,
            reranker_type="cosine",
        )
        retriever = DartboardRetriever(config, embedding_model)

        result = retriever.retrieve(query, dataset.chunks)

        # Analyze diversity
        diversity_stats = diversity_analyzer.analyze_diversity(result)

        # Count unique base passages
        base_ids = set()
        for chunk in result.chunks:
            base_id = chunk.metadata.get("base_id")
            if base_id is not None:
                base_ids.add(base_id)

        print(f"  Retrieved: {len(result.chunks)} passages")
        print(f"  Unique base passages: {len(base_ids)}/10 possible")
        print(
            f"  Diversity (mean pairwise distance): {diversity_stats['mean_distance']:.4f}"
        )
        print(f"  Min distance (closest pair): {diversity_stats['min_distance']:.4f}")
        print(f"  Max distance (farthest pair): {diversity_stats['max_distance']:.4f}")

        # Show a few retrieved passages
        print(f"\n  First 5 retrieved:")
        for i, chunk in enumerate(result.chunks[:5], 1):
            base_id = chunk.metadata.get("base_id", "?")
            copy_num = chunk.metadata.get("copy", "?")
            score = result.scores[i - 1]
            print(f"    {i}. Base:{base_id} Copy:{copy_num} Score:{score:.4f}")

        print()

    print("=" * 70)
    print("📊 Analysis:")
    print("=" * 70)
    print()
    print("Higher σ values should select more diverse passages (more unique base_ids)")
    print("Lower σ values focus more on relevance, potentially selecting duplicates")
    print()
    print("✓ Redundancy test complete!")


if __name__ == "__main__":
    main()
