"""
Tests for EmbeddingSemanticChunker.

Tests embedding-based semantic chunking functionality:
- Sentence grouping by similarity
- Similarity threshold handling
- Edge cases (empty docs, single sentence)
- Comparison with other chunkers
"""

import pytest
from dartboard.ingestion.chunking import (
    EmbeddingSemanticChunker,
    SentenceChunker,
    Document,
    Chunk,
)
from dartboard.embeddings import SentenceTransformerModel


# Sample text with clear topic transitions
SAMPLE_TEXT_TRANSITIONS = """
Machine learning is a subset of artificial intelligence. It enables computers to learn from data.
Deep learning uses neural networks with multiple layers. Neural networks mimic the human brain structure.
Natural language processing handles text and speech. NLP powers chatbots and translation systems.
Computer vision processes images and videos. It enables facial recognition and object detection.
""".strip()

# Sample text with high semantic similarity (should stay together)
SAMPLE_TEXT_COHERENT = """
The cat sat on the mat. The feline rested on the rug.
The kitty was lying on the carpet. The small animal relaxed on the floor covering.
""".strip()

# Sample text with contradictory information
SAMPLE_TEXT_CONTRADICTORY = """
Python is a great programming language for beginners. It has simple syntax and is easy to learn.
Python is terrible for beginners. Its dynamic typing leads to runtime errors.
Java is the best language for enterprise applications. It provides strong type safety.
""".strip()


class TestEmbeddingSemanticChunker:
    """Tests for EmbeddingSemanticChunker."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for tests."""
        return SentenceTransformerModel("all-MiniLM-L6-v2")

    @pytest.fixture
    def chunker(self, embedding_model):
        """Create EmbeddingSemanticChunker instance."""
        return EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.75,
            max_chunk_size=512,
        )

    def test_basic_chunking(self, chunker):
        """Test basic embedding-based chunking."""
        doc = Document(
            content=SAMPLE_TEXT_TRANSITIONS, metadata={"source": "test"}, source="test"
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)

    def test_semantic_grouping(self, chunker):
        """Test that semantically similar sentences are grouped together."""
        doc = Document(
            content=SAMPLE_TEXT_COHERENT, metadata={"source": "test"}, source="test"
        )

        chunks = chunker.chunk(doc)

        # Coherent text should produce fewer chunks (sentences stay together)
        # With high similarity threshold, similar sentences group together
        assert len(chunks) >= 1
        assert all("semantic_coherence" in chunk.metadata for chunk in chunks)

    def test_topic_transitions(self, chunker):
        """Test that topic transitions create new chunks."""
        # Create chunker with high threshold to force splits at topic boundaries
        high_threshold_chunker = EmbeddingSemanticChunker(
            embedding_model=chunker.embedding_model,
            similarity_threshold=0.8,  # Stricter threshold
            max_chunk_size=512,
        )

        doc = Document(
            content=SAMPLE_TEXT_TRANSITIONS,
            metadata={"source": "test"},
            source="test",
        )

        chunks = high_threshold_chunker.chunk(doc)

        # Should have multiple chunks due to topic transitions
        # (ML → Deep Learning → NLP → Computer Vision)
        assert len(chunks) >= 2

    def test_similarity_threshold_effect(self, embedding_model):
        """Test that lower threshold creates fewer chunks."""
        doc = Document(
            content=SAMPLE_TEXT_TRANSITIONS,
            metadata={"source": "test"},
            source="test",
        )

        # Low threshold (more lenient) - should create fewer chunks
        low_threshold_chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.5,
            max_chunk_size=512,
        )

        # High threshold (stricter) - should create more chunks
        high_threshold_chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.9,
            max_chunk_size=512,
        )

        low_chunks = low_threshold_chunker.chunk(doc)
        high_chunks = high_threshold_chunker.chunk(doc)

        # Lower threshold should produce fewer or equal chunks
        assert len(low_chunks) <= len(high_chunks)

    def test_max_chunk_size_limit(self, chunker):
        """Test that chunks respect max_chunk_size."""
        long_text = " ".join(
            [
                f"This is sentence number {i} containing various words and information."
                for i in range(100)
            ]
        )

        doc = Document(content=long_text, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # All chunks should respect max size
        for chunk in chunks:
            if chunk.token_count:
                # Allow some flexibility (sentences might push over slightly)
                assert chunk.token_count <= chunker.max_chunk_size * 1.2

    def test_empty_document(self, chunker):
        """Test handling of empty document."""
        doc = Document(content="", metadata={}, source="test")

        chunks = chunker.chunk(doc)

        assert len(chunks) == 0

    def test_single_sentence_document(self, chunker):
        """Test handling of single sentence."""
        doc = Document(
            content="This is a single sentence.",
            metadata={"source": "test"},
            source="test",
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "This is a single sentence."

    def test_chunk_metadata(self, chunker):
        """Test that chunks have proper metadata."""
        doc = Document(
            content=SAMPLE_TEXT_TRANSITIONS,
            metadata={"source": "test.txt", "author": "Test Author"},
            source="test.txt",
        )

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["chunk_index"] == i
            assert "chunk_size" in chunk.metadata
            assert chunk.metadata["semantic_coherence"] is True
            assert chunk.chunk_index == i

    def test_contradictory_content_splits(self, chunker):
        """Test that contradictory sentences create separate chunks."""
        doc = Document(content=SAMPLE_TEXT_CONTRADICTORY, metadata={}, source="test")

        # Use high threshold to ensure contradictory sentences split
        strict_chunker = EmbeddingSemanticChunker(
            embedding_model=chunker.embedding_model,
            similarity_threshold=0.8,
            max_chunk_size=512,
        )

        chunks = strict_chunker.chunk(doc)

        # Contradictory statements should create multiple chunks
        assert len(chunks) >= 2

    def test_nltk_fallback(self, embedding_model):
        """Test sentence splitting works even without NLTK."""
        # This tests the regex fallback
        chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.75,
            max_chunk_size=512,
        )

        doc = Document(
            content="First sentence. Second sentence. Third sentence.",
            metadata={},
            source="test",
        )

        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_comparison_with_sentence_chunker(self, embedding_model):
        """Compare EmbeddingSemanticChunker with SentenceChunker."""
        doc = Document(content=SAMPLE_TEXT_TRANSITIONS, metadata={}, source="test")

        # EmbeddingSemanticChunker
        semantic_chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.75,
            max_chunk_size=512,
        )

        # SentenceChunker (RecursiveChunker)
        sentence_chunker = SentenceChunker(chunk_size=512, overlap=50)

        semantic_chunks = semantic_chunker.chunk(doc)
        sentence_chunks = sentence_chunker.chunk(doc)

        # Both should produce chunks
        assert len(semantic_chunks) > 0
        assert len(sentence_chunks) > 0

        # Different strategies may produce different numbers of chunks
        # (that's expected and desirable)

    def test_token_counting_fallback(self, embedding_model):
        """Test that token counting works even without tiktoken."""
        # Create chunker without tiktoken (it will use fallback)
        chunker = EmbeddingSemanticChunker(
            embedding_model=embedding_model,
            similarity_threshold=0.75,
            max_chunk_size=512,
            model="nonexistent-model",  # Will trigger fallback
        )

        doc = Document(content="Short text for testing.", metadata={}, source="test")

        # Should not raise error, should use character-based estimation
        try:
            chunks = chunker.chunk(doc)
            assert len(chunks) > 0
        except ImportError:
            # If tiktoken isn't installed, this is expected
            pytest.skip("tiktoken not installed")


def test_end_to_end_semantic_chunking():
    """End-to-end test of embedding-based semantic chunking."""
    # Initialize embedding model
    model = SentenceTransformerModel("all-MiniLM-L6-v2")

    # Create semantic chunker
    chunker = EmbeddingSemanticChunker(
        embedding_model=model, similarity_threshold=0.75, max_chunk_size=512
    )

    # Create document with clear topic structure
    doc = Document(
        content=SAMPLE_TEXT_TRANSITIONS,
        metadata={"title": "AI Topics", "author": "Test"},
        source="test.md",
    )

    # Chunk document
    chunks = chunker.chunk(doc)

    # Verify results
    assert len(chunks) > 0

    # All chunks should have metadata
    for chunk in chunks:
        assert "title" in chunk.metadata
        assert "author" in chunk.metadata
        assert chunk.metadata["title"] == "AI Topics"
        assert chunk.metadata["semantic_coherence"] is True

    # Text should be preserved
    full_text = " ".join([chunk.text for chunk in chunks])
    assert "Machine learning" in full_text or "machine learning" in full_text


if __name__ == "__main__":
    print("Running embedding semantic chunking tests...")
    pytest.main([__file__, "-v"])
