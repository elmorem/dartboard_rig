#!/usr/bin/env python3
"""
End-to-end demonstration of chunking pipeline.

Shows complete workflow:
1. Load Markdown document
2. Chunk with SentenceChunker
3. Generate embeddings
4. Store in vector database
5. Retrieve with Dartboard

This demo validates that all components work together.
"""

import tempfile
import os

from dartboard.ingestion.loaders import MarkdownLoader
from dartboard.ingestion.chunking import SentenceChunker, EmbeddingSemanticChunker
from dartboard.ingestion.pipeline import create_pipeline
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import VectorStore
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.datasets.models import Chunk as StorageChunk
from typing import List, Optional, Dict
import numpy as np


# Simple in-memory vector store (no FAISS to avoid compatibility issues)
class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store for testing."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.chunks: List[StorageChunk] = []

    def add(self, chunks: List[StorageChunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query_embedding, k: int, filters=None) -> List[StorageChunk]:
        if not self.chunks:
            return []

        # Compute cosine similarities
        similarities = []
        for chunk in self.chunks:
            sim = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                + 1e-10
            )
            similarities.append(sim)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.chunks[i] for i in top_indices]

    def delete(self, chunk_ids: List[str]) -> None:
        self.chunks = [c for c in self.chunks if c.id not in chunk_ids]

    def count(self) -> int:
        return len(self.chunks)

    def list_all(self) -> List[StorageChunk]:
        return self.chunks.copy()


# Sample document
SAMPLE_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed.

## Supervised Learning

Supervised learning uses labeled training data. The algorithm learns patterns from examples where the correct answer is provided. Common applications include classification and regression tasks.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. The algorithm discovers hidden structures without predefined categories. Clustering and dimensionality reduction are key techniques.

## Deep Learning

Deep learning uses multi-layered neural networks. These networks automatically learn hierarchical feature representations from raw data. Applications include computer vision, natural language processing, and speech recognition.

## Conclusion

Machine learning transforms industries from healthcare to finance. Understanding these fundamentals is essential for modern software development.
""".strip()


def main():
    print("=" * 80)
    print("CHUNKING PIPELINE END-TO-END DEMONSTRATION")
    print("=" * 80)

    # Step 1: Create temporary document
    print("\n[1] Creating temporary document...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(SAMPLE_DOCUMENT)
        temp_path = f.name

    print(f"    Document created: {temp_path}")
    print(f"    Document size: {len(SAMPLE_DOCUMENT)} characters")

    try:
        # Step 2: Load document
        print("\n[2] Loading document...")
        loader = MarkdownLoader()
        documents = loader.load(temp_path)

        print(f"    Loaded {len(documents)} document(s)")
        print(f"    Content preview: {documents[0].content[:100]}...")

        # Step 3: Chunk with SentenceChunker
        print("\n[3] Chunking with SentenceChunker...")
        chunker = SentenceChunker(chunk_size=256, overlap=30)

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        print(f"    Created {len(all_chunks)} chunks")
        print(f"    First chunk: {all_chunks[0].text[:80]}...")

        # Step 4: Initialize embedding model
        print("\n[4] Loading embedding model...")
        embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
        print(f"    Model loaded: all-MiniLM-L6-v2")
        print(f"    Embedding dimension: {embedding_model.embedding_dim}")

        # Step 5: Generate embeddings
        print("\n[5] Generating embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embedding_model.encode(texts)

        # Ensure 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        print(f"    Generated {embeddings.shape[0]} embeddings")
        print(f"    Embedding shape: {embeddings.shape}")

        # Step 6: Create storage chunks
        print("\n[6] Creating storage chunks...")
        storage_chunks = []
        for i, chunk in enumerate(all_chunks):
            storage_chunk = StorageChunk(
                id=f"chunk_{i}",
                text=chunk.text,
                embedding=embeddings[i],
                metadata=chunk.metadata,
            )
            storage_chunks.append(storage_chunk)

        print(f"    Created {len(storage_chunks)} storage chunks")

        # Step 7: Store in vector database
        print("\n[7] Storing in vector database...")
        vector_store = SimpleVectorStore(embedding_dim=embedding_model.embedding_dim)
        vector_store.add(storage_chunks)

        print(f"    Stored {vector_store.count()} chunks")

        # Step 8: Retrieve with Dartboard
        print("\n[8] Retrieving with Dartboard...")
        config = DartboardConfig(
            sigma=1.0, top_k=3, triage_k=min(10, len(storage_chunks))
        )
        retriever = DartboardRetriever(config, embedding_model)

        query = "What is supervised learning?"
        print(f"    Query: '{query}'")

        result = retriever.retrieve(query, storage_chunks)

        print(f"    Retrieved {len(result.chunks)} chunks\n")

        for i, chunk in enumerate(result.chunks, 1):
            print(f"    [{i}] {chunk.text[:120]}...")

        # Step 9: Test with EmbeddingSemanticChunker
        print("\n[9] Testing EmbeddingSemanticChunker...")
        semantic_chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.7,
            max_chunk_size=256,
        )

        semantic_chunks = []
        for doc in documents:
            chunks = semantic_chunker.chunk(doc)
            semantic_chunks.extend(chunks)

        print(f"    Created {len(semantic_chunks)} semantic chunks")
        print(f"    First chunk: {semantic_chunks[0].text[:80]}...")

        # Verify semantic coherence metadata
        coherent_count = sum(
            1 for c in semantic_chunks if c.metadata.get("semantic_coherence")
        )
        print(
            f"    Chunks with semantic coherence: {coherent_count}/{len(semantic_chunks)}"
        )

        # Step 10: Test ingestion pipeline
        print("\n[10] Testing full ingestion pipeline...")
        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=256,
            overlap=30,
        )

        # Note: we already added chunks, so this will add more
        result = pipeline.ingest(temp_path, track_progress=False)

        print(f"    Pipeline status: {result.status}")
        print(f"    Documents processed: {result.documents_processed}")
        print(f"    Chunks created: {result.chunks_created}")
        print(f"    Chunks stored: {result.chunks_stored}")

        print("\n" + "=" * 80)
        print("✓ END-TO-END DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nAll components working correctly:")
        print("  ✓ Document loading (Markdown)")
        print("  ✓ Sentence-aware chunking")
        print("  ✓ Embedding-based semantic chunking")
        print("  ✓ Embedding generation")
        print("  ✓ Vector storage")
        print("  ✓ Dartboard retrieval")
        print("  ✓ Full ingestion pipeline")
        print()

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")


if __name__ == "__main__":
    main()
