"""
End-to-end integration tests for chunking pipeline.

Tests the complete workflow:
1. Load document (PDF/Markdown)
2. Chunk into passages
3. Generate embeddings
4. Store in vector database
5. Retrieve with Dartboard
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Dict

from dartboard.ingestion.loaders import PDFLoader, MarkdownLoader
from dartboard.ingestion.chunking import (
    SentenceChunker,
    EmbeddingSemanticChunker,
    Document,
)
from dartboard.ingestion.pipeline import IngestionPipeline, create_pipeline
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore, VectorStore
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.datasets.models import Chunk as StorageChunk
from dartboard.config import get_embedding_config


class SimpleInMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing (no FAISS)."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.chunks: List[StorageChunk] = []

    def add(self, chunks: List[StorageChunk]) -> None:
        """Add chunks to store."""
        self.chunks.extend(chunks)

    def search(self, query_embedding, k: int, filters=None) -> List[StorageChunk]:
        """Simple cosine similarity search."""
        if not self.chunks:
            return []

        import numpy as np

        # Compute similarities
        similarities = []
        for chunk in self.chunks:
            sim = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                + 1e-10
            )
            similarities.append(sim)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.chunks[i] for i in top_indices]

    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID."""
        self.chunks = [c for c in self.chunks if c.id not in chunk_ids]

    def count(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def list_all(self) -> List[StorageChunk]:
        """Return all chunks."""
        return self.chunks.copy()


# Sample markdown content for testing
SAMPLE_MARKDOWN = """
# Machine Learning Basics

## Introduction

Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning

Supervised learning uses labeled data to train models. The algorithm learns from examples where the correct answer is provided.

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering and dimensionality reduction are common applications.

## Deep Learning

Deep learning uses neural networks with multiple layers. These networks can automatically learn hierarchical features from raw data.

### Applications

Deep learning powers computer vision, natural language processing, and speech recognition systems.

## Conclusion

Machine learning continues to transform industries from healthcare to finance. Understanding its fundamentals is essential for modern developers.
""".strip()


class TestEndToEndChunkingPipeline:
    """End-to-end integration tests for chunking pipeline."""

    # Note: embedding_model fixture provided by conftest.py

    @pytest.fixture
    def vector_store(self, embedding_model):
        """Create simple in-memory vector store."""
        return SimpleInMemoryVectorStore(embedding_dim=embedding_model.embedding_dim)

    @pytest.fixture
    def temp_markdown_file(self):
        """Create temporary markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(SAMPLE_MARKDOWN)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_markdown_to_chunks_to_retrieval(
        self, temp_markdown_file, embedding_model, vector_store
    ):
        """Test complete pipeline: Markdown → Chunks → Embeddings → Retrieval."""
        # Step 1: Load markdown
        loader = MarkdownLoader()
        documents = loader.load(temp_markdown_file)

        assert len(documents) == 1
        assert "Machine Learning" in documents[0].content

        # Step 2: Chunk document
        chunker = SentenceChunker(chunk_size=256, overlap=30)
        all_chunks = []

        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0
        print(f"Created {len(all_chunks)} chunks")

        # Step 3: Generate embeddings
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embedding_model.encode(texts)

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        assert embeddings.shape[0] == len(all_chunks)
        assert embeddings.shape[1] == embedding_model.embedding_dim

        # Step 4: Create storage chunks and add to vector store
        storage_chunks = []
        for i, chunk in enumerate(all_chunks):
            storage_chunk = StorageChunk(
                id=f"chunk_{i}",
                text=chunk.text,
                embedding=embeddings[i],
                metadata=chunk.metadata,
            )
            storage_chunks.append(storage_chunk)

        vector_store.add(storage_chunks)

        # Step 5: Retrieve with Dartboard
        config = DartboardConfig(sigma=1.0, top_k=3, triage_k=10)
        retriever = DartboardRetriever(config, embedding_model)

        query = "What is supervised learning?"
        result = retriever.retrieve(query, storage_chunks)

        # Verify retrieval
        assert len(result.chunks) == 3
        assert any("supervised" in chunk.text.lower() for chunk in result.chunks)

    def test_pipeline_with_sentence_chunker(
        self, temp_markdown_file, embedding_model, vector_store
    ):
        """Test ingestion pipeline with SentenceChunker."""
        # Create pipeline
        loader = MarkdownLoader()
        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=256,
            overlap=30,
        )

        # Ingest document
        result = pipeline.ingest(temp_markdown_file, track_progress=True)

        # Verify ingestion
        assert result.status == "success"
        assert result.documents_processed == 1
        assert result.chunks_created > 0
        assert result.chunks_stored > 0

        print(f"Ingested {result.chunks_created} chunks")

        # Verify chunks are in vector store
        all_chunks = vector_store.list_all()
        assert len(all_chunks) == result.chunks_stored

    def test_pipeline_with_embedding_semantic_chunker(
        self, temp_markdown_file, embedding_model, vector_store
    ):
        """Test pipeline with EmbeddingSemanticChunker."""
        # Load document
        loader = MarkdownLoader()
        documents = loader.load(temp_markdown_file)

        # Chunk with embedding-based semantic chunking
        chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.7,
            max_chunk_size=256,
        )

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0
        print(f"Created {len(all_chunks)} semantic chunks")

        # Verify semantic coherence metadata
        assert all(chunk.metadata.get("semantic_coherence") for chunk in all_chunks)

    def test_comparison_sentence_vs_semantic_chunking(
        self, temp_markdown_file, embedding_model
    ):
        """Compare SentenceChunker vs EmbeddingSemanticChunker."""
        # Load document
        loader = MarkdownLoader()
        documents = loader.load(temp_markdown_file)
        doc = documents[0]

        # Chunk with SentenceChunker
        sentence_chunker = SentenceChunker(chunk_size=256, overlap=30)
        sentence_chunks = sentence_chunker.chunk(doc)

        # Chunk with EmbeddingSemanticChunker
        semantic_chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.75,
            max_chunk_size=256,
        )
        semantic_chunks = semantic_chunker.chunk(doc)

        # Both should produce chunks
        assert len(sentence_chunks) > 0
        assert len(semantic_chunks) > 0

        print(f"SentenceChunker: {len(sentence_chunks)} chunks")
        print(f"EmbeddingSemanticChunker: {len(semantic_chunks)} chunks")

        # Different strategies may produce different numbers
        # This is expected and shows they work differently

    def test_query_retrieval_quality(
        self, temp_markdown_file, embedding_model, vector_store
    ):
        """Test retrieval quality with different queries."""
        # Setup pipeline
        loader = MarkdownLoader()
        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=256,
            overlap=30,
        )

        # Ingest document
        result = pipeline.ingest(temp_markdown_file)
        assert result.status == "success"

        # Get all chunks
        all_chunks = vector_store.list_all()

        # Create retriever
        config = DartboardConfig(sigma=1.0, top_k=3, triage_k=min(10, len(all_chunks)))
        retriever = DartboardRetriever(config, embedding_model)

        # Test multiple queries
        queries = [
            "What is machine learning?",
            "Explain supervised learning",
            "What are applications of deep learning?",
            "What is unsupervised learning?",
        ]

        for query in queries:
            result = retriever.retrieve(query, all_chunks)

            # Should return requested number of chunks
            assert len(result.chunks) == min(3, len(all_chunks))

            # Results should be diverse (Dartboard property)
            if len(result.chunks) >= 2:
                # Check that chunks are not identical
                texts = [chunk.text for chunk in result.chunks]
                assert len(set(texts)) == len(texts), "Chunks should be unique"

            print(f"\nQuery: {query}")
            print(f"Retrieved {len(result.chunks)} chunks")

    def test_batch_ingestion(self, embedding_model, vector_store):
        """Test batch ingestion of multiple documents."""
        # Create multiple temporary markdown files
        temp_files = []

        for i in range(3):
            content = f"""
# Document {i + 1}

This is document number {i + 1}. It contains unique content about topic {i + 1}.

## Section A

Content for section A in document {i + 1}.

## Section B

Content for section B in document {i + 1}.
""".strip()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(content)
                temp_files.append(f.name)

        try:
            # Create pipeline
            loader = MarkdownLoader()
            pipeline = create_pipeline(
                loader=loader,
                embedding_model=embedding_model,
                vector_store=vector_store,
                chunk_size=256,
                overlap=30,
            )

            # Batch ingest
            results = pipeline.ingest_batch(temp_files, track_progress=True)

            # Verify all documents processed
            assert len(results) == 3
            assert all(r.status == "success" for r in results)

            total_chunks = sum(r.chunks_created for r in results)
            assert total_chunks > 0

            print(
                f"Batch ingested {total_chunks} chunks from {len(temp_files)} documents"
            )

        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def test_error_handling_missing_file(self, embedding_model, vector_store):
        """Test pipeline handles missing files gracefully."""
        loader = MarkdownLoader()
        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # Try to ingest non-existent file
        result = pipeline.ingest("/nonexistent/file.md")

        # Should fail gracefully
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_empty_document_handling(self, embedding_model, vector_store):
        """Test pipeline handles empty documents."""
        # Create empty markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            loader = MarkdownLoader()
            pipeline = create_pipeline(
                loader=loader,
                embedding_model=embedding_model,
                vector_store=vector_store,
            )

            result = pipeline.ingest(temp_path)

            # Should complete but create no chunks
            # (behavior depends on implementation - could be success or failed)
            if result.status == "success":
                assert result.chunks_created == 0

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def test_full_rag_workflow():
    """Test complete RAG workflow from document to answer."""
    # Initialize components
    config = get_embedding_config()
    embedding_model = SentenceTransformerModel(config.model_name)
    vector_store = SimpleInMemoryVectorStore(
        embedding_dim=embedding_model.embedding_dim
    )

    # Create temporary document
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(SAMPLE_MARKDOWN)
        temp_path = f.name

    try:
        # Step 1: Ingest document
        loader = MarkdownLoader()
        pipeline = create_pipeline(
            loader=loader,
            embedding_model=embedding_model,
            vector_store=vector_store,
            chunk_size=256,
            overlap=30,
        )

        result = pipeline.ingest(temp_path, track_progress=True)
        assert result.status == "success"

        print(f"\n=== Ingestion Complete ===")
        print(f"Documents: {result.documents_processed}")
        print(f"Chunks: {result.chunks_created}")

        # Step 2: Retrieve relevant passages
        all_chunks = vector_store.list_all()

        config = DartboardConfig(sigma=1.0, top_k=5, triage_k=min(20, len(all_chunks)))
        retriever = DartboardRetriever(config, embedding_model)

        query = "What are the different types of machine learning?"
        retrieval_result = retriever.retrieve(query, all_chunks)

        print(f"\n=== Query: {query} ===")
        print(f"Retrieved {len(retrieval_result.chunks)} chunks\n")

        for i, chunk in enumerate(retrieval_result.chunks, 1):
            print(f"[{i}] {chunk.text[:100]}...")

        # Step 3: Verify quality
        # Should retrieve relevant chunks about ML types
        retrieved_texts = " ".join([c.text.lower() for c in retrieval_result.chunks])

        # At least one chunk should mention supervised or unsupervised learning
        assert (
            "supervised" in retrieved_texts or "unsupervised" in retrieved_texts
        ), "Should retrieve chunks about ML types"

        print("\n=== RAG Workflow Complete ===")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    print("Running chunking pipeline integration tests...")
    pytest.main([__file__, "-v", "-s"])
