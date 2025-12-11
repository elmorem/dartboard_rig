"""
Test Dartboard on highly diverse datasets.

Compares behavior across different diversity targets to validate
that Dartboard naturally balances relevance and diversity.
"""

from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.synthetic import SyntheticConfig, SyntheticDatasetGenerator
from dartboard.evaluation.metrics import DiversityAnalyzer


def main():
    print("=" * 70)
    print("🌈 Dartboard Diversity Test")
    print("=" * 70)
    print()

    # Initialize embedding model
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")
    print()

    # Test different diversity targets
    diversity_targets = [0.5, 0.7, 0.9]

    analyzer = DiversityAnalyzer()

    for diversity_target in diversity_targets:
        print("=" * 70)
        print(f"Diversity Target: {diversity_target}")
        print("=" * 70)
        print()

        # Generate diverse dataset
        print(f"🔨 Generating dataset with diversity target {diversity_target}...")
        synthetic_config = SyntheticConfig(
            num_clusters=2,
            passages_per_cluster=10,
            embedding_dim=get_embedding_config().embedding_dim,
            seed=42,
        )
        generator = SyntheticDatasetGenerator(synthetic_config)
        dataset = generator.generate_diverse_dataset(diversity_target=diversity_target)

        print(f"✓ Generated {len(dataset.chunks)} passages")
        print()

        # Analyze corpus diversity
        corpus_embeddings = [c.embedding for c in dataset.chunks]
        import numpy as np

        corpus_distances = []
        for i in range(min(10, len(corpus_embeddings))):
            for j in range(i + 1, min(10, len(corpus_embeddings))):
                from dartboard.utils import cosine_distance

                dist = cosine_distance(corpus_embeddings[i], corpus_embeddings[j])
                corpus_distances.append(dist)

        print(f"📊 Corpus Statistics (first 10 passages):")
        print(f"   Mean pairwise distance: {np.mean(corpus_distances):.4f}")
        print(f"   Min distance: {np.min(corpus_distances):.4f}")
        print(f"   Max distance: {np.max(corpus_distances):.4f}")
        print()

        # Test retrieval with different sigma values
        print("🎯 Testing retrieval with σ=1.0...")
        config = DartboardConfig(
            sigma=1.0, top_k=5, triage_k=15, reranker_type="cosine"
        )
        retriever = DartboardRetriever(config, embedding_model)

        query = dataset.queries[0]
        result = retriever.retrieve(query, dataset.chunks)

        # Analyze retrieved diversity
        diversity_stats = analyzer.analyze_diversity(result)

        print(f"   Retrieved {len(result.chunks)} passages")
        print(f"   Mean distance: {diversity_stats['mean_distance']:.4f}")
        print(f"   Min distance: {diversity_stats['min_distance']:.4f}")
        print(f"   Max distance: {diversity_stats['max_distance']:.4f}")
        print()

    print("=" * 70)
    print("📊 Analysis:")
    print("=" * 70)
    print()
    print("Higher diversity targets in corpus should lead to:")
    print("- Higher mean pairwise distances in corpus")
    print("- Dartboard should maintain high diversity in retrieval")
    print()
    print("✓ Diversity test complete!")


if __name__ == "__main__":
    main()
