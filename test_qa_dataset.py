"""
Test text-based Q&A dataset generation and retrieval.

This tests Dartboard on realistic text data with semantic topics.
"""

from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.synthetic import TextBasedSyntheticGenerator
from dartboard.evaluation.metrics import RetrievalEvaluator


def main():
    print("=" * 70)
    print("📚 Text-Based Q&A Dataset Test")
    print("=" * 70)
    print()

    # Initialize embedding model
    print("📦 Loading embedding model...")
    embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})")
    print()

    # Generate text-based dataset
    print("🔨 Generating Q&A dataset...")
    topics = [
        "machine learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "deep learning",
    ]

    generator = TextBasedSyntheticGenerator(embedding_model)
    dataset = generator.generate_qa_dataset(topics, passages_per_topic=5)

    print(f"✓ Generated dataset:")
    print(f"   Topics: {len(topics)}")
    print(f"   Passages: {len(dataset.chunks)}")
    print(f"   Queries: {len(dataset.queries)}")
    print()

    # Show sample passages
    print("📄 Sample passages:")
    for topic in topics[:2]:
        topic_chunks = [c for c in dataset.chunks if c.metadata.get("topic") == topic]
        print(f"\n  Topic: {topic}")
        print(f"  - {topic_chunks[0].text}")
        if len(topic_chunks) > 1:
            print(f"  - {topic_chunks[1].text}")

    print()
    print("=" * 70)

    # Test retrieval
    print("\n🎯 Testing Dartboard retrieval...")
    config = DartboardConfig(sigma=1.0, top_k=5, triage_k=20, reranker_type="cosine")
    retriever = DartboardRetriever(config, embedding_model)

    # Test on first query
    query_idx = 0
    query = dataset.queries[query_idx]
    print(f"\nQuery: '{query}'")
    print("-" * 70)

    result = retriever.retrieve(query, dataset.chunks)

    print(f"\nTop {len(result.chunks)} results:")
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores), 1):
        topic = chunk.metadata.get("topic", "unknown")
        print(f"\n{i}. [Score: {score:.4f}, Topic: {topic}]")
        print(f"   {chunk.text}")

    # Evaluate all queries
    print()
    print("=" * 70)
    print("\n📊 Full Evaluation:")
    print("-" * 70)

    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    # Retrieve for all queries
    all_results = []
    for query in dataset.queries:
        result = retriever.retrieve(query, dataset.chunks)
        all_results.append(result)

    # Evaluate
    report = evaluator.evaluate_dataset(all_results, dataset)

    print(f"\nDataset: {report.dataset_name}")
    print(f"NDCG:         {report.metrics['ndcg']:.4f}")
    print(f"MAP:          {report.metrics['map']:.4f}")
    print(f"Precision@1:  {report.metrics['precision@1']:.4f}")
    print(f"Precision@3:  {report.metrics['precision@3']:.4f}")
    print(f"Precision@5:  {report.metrics['precision@5']:.4f}")
    print(f"Recall@5:     {report.metrics['recall@5']:.4f}")
    print(f"Diversity:    {report.metrics['diversity']:.4f}")

    print()
    print("=" * 70)
    print("✓ Text-based Q&A test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
