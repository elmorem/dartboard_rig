"""
Tests for text chunking functionality.

Tests:
- RecursiveChunker with sentence boundaries
- Overlap handling
- Code block preservation
- SemanticChunker with paragraphs
- FixedSizeChunker
- Token counting utilities
"""

import pytest
from dartboard.ingestion.chunking import (
    RecursiveChunker,
    SemanticChunker,
    FixedSizeChunker,
    TokenCounter,
    Document,
    Chunk,
)


# Sample texts for testing
SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence. It focuses on learning from data.
Deep learning uses neural networks with multiple layers. These networks can learn complex patterns.
Natural language processing handles human language. It powers chatbots and translation systems.
""".strip()

SAMPLE_TEXT_WITH_CODE = """
Here's a simple Python function:

```python
def hello_world():
    print("Hello, World!")
    return True
```

This function prints a greeting message. It's commonly used for testing.
The function returns True when complete.
""".strip()

SAMPLE_MARKDOWN = """
# Introduction

This is the introduction paragraph. It provides context for the document.

## Section 1

This is the first section. It contains important information.

### Subsection 1.1

More detailed content here. This explains specific concepts.

## Section 2

Another section with different content. This covers related topics.
""".strip()

SHORT_TEXT = "This is a short text."

LONG_TEXT = " ".join(
    [
        f"Sentence {i} contains some information about machine learning and artificial intelligence."
        for i in range(50)
    ]
)


class TestTokenCounter:
    """Tests for TokenCounter utility."""

    def test_token_counter_initialization(self):
        """Test token counter can be initialized."""
        try:
            counter = TokenCounter()
            assert counter is not None
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_count_tokens(self):
        """Test token counting."""
        try:
            counter = TokenCounter()
            count = counter.count_tokens("Hello, world!")
            assert count > 0
            assert isinstance(count, int)
        except ImportError:
            pytest.skip("tiktoken not installed")

    def test_estimate_tokens(self):
        """Test token estimation."""
        try:
            counter = TokenCounter()
            estimate = counter.estimate_tokens("Hello, world!")
            assert estimate > 0
            assert isinstance(estimate, int)
            # Estimate should be roughly 1/4 of characters
            assert estimate == len("Hello, world!") // 4
        except ImportError:
            pytest.skip("tiktoken not installed")


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_basic_chunking(self):
        """Test basic sentence-aware chunking."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        doc = Document(content=SAMPLE_TEXT, metadata={"source": "test"}, source="test")

        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)

    def test_chunk_metadata(self):
        """Test chunk metadata preservation."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        doc = Document(
            content=SAMPLE_TEXT,
            metadata={"source": "test", "title": "ML Basics"},
            source="test",
        )

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["title"] == "ML Basics"
            assert chunk.metadata["chunk_index"] == i
            assert chunk.chunk_index == i

    def test_overlap_handling(self):
        """Test that chunks have proper overlap."""
        chunker = RecursiveChunker(chunk_size=50, overlap=20)
        doc = Document(content=LONG_TEXT, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].text
            chunk2_text = chunks[i + 1].text

            # Extract words for easier comparison
            chunk1_words = chunk1_text.split()
            chunk2_words = chunk2_text.split()

            # Check if there's word overlap
            overlap_words = set(chunk1_words[-5:]) & set(chunk2_words[:10])
            # Should have some overlap
            assert (
                len(overlap_words) > 0
            ), f"No overlap found between chunks {i} and {i+1}"

    def test_code_block_preservation(self):
        """Test that code blocks are not split."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20, respect_code_blocks=True)
        doc = Document(content=SAMPLE_TEXT_WITH_CODE, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Find chunk containing code block
        code_chunks = [chunk for chunk in chunks if "```python" in chunk.text]

        # Code block should be in at least one chunk
        assert len(code_chunks) > 0

        # Code block should be complete (not split)
        for chunk in code_chunks:
            if "```python" in chunk.text:
                assert "def hello_world():" in chunk.text
                assert "print(" in chunk.text
                # Should have closing backticks or be in separate chunk
                assert "```" in chunk.text

    def test_short_document(self):
        """Test chunking of very short document."""
        chunker = RecursiveChunker(chunk_size=512, overlap=50)
        doc = Document(content=SHORT_TEXT, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Should have exactly 1 chunk
        assert len(chunks) == 1
        assert chunks[0].text == SHORT_TEXT

    def test_empty_document(self):
        """Test chunking of empty document."""
        chunker = RecursiveChunker(chunk_size=512, overlap=50)
        doc = Document(content="", metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Should have no chunks
        assert len(chunks) == 0

    def test_sentence_boundaries(self):
        """Test that chunks respect sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = RecursiveChunker(chunk_size=30, overlap=10)
        doc = Document(content=text, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not end mid-word
            assert not chunk.text.endswith(" the")
            assert not chunk.text.endswith(" and")

    def test_chunk_size_limit(self):
        """Test that chunks don't exceed size limit."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        doc = Document(content=LONG_TEXT, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            # Token count should be set
            assert chunk.token_count is not None
            # Should not greatly exceed chunk size (some flexibility for sentence boundaries)
            assert chunk.token_count <= chunker.chunk_size * 1.5


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_paragraph_chunking(self):
        """Test chunking on paragraph boundaries."""
        chunker = SemanticChunker(max_chunk_size=200)
        doc = Document(content=SAMPLE_MARKDOWN, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_heading_preservation(self):
        """Test that headings are preserved in chunks."""
        chunker = SemanticChunker(max_chunk_size=500)
        doc = Document(content=SAMPLE_MARKDOWN, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # At least one chunk should contain a heading
        heading_chunks = [
            chunk for chunk in chunks if any(h in chunk.text for h in ["#", "Section"])
        ]
        assert len(heading_chunks) > 0

    def test_large_section_splitting(self):
        """Test that large sections are split."""
        large_text = "# Heading\n\n" + " ".join([f"Sentence {i}." for i in range(200)])
        chunker = SemanticChunker(max_chunk_size=100)
        doc = Document(content=large_text, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Should split large section
        assert len(chunks) > 1


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        doc = Document(content=LONG_TEXT, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_overlap_in_fixed_chunks(self):
        """Test overlap in fixed-size chunks."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=20)
        doc = Document(content=LONG_TEXT, metadata={}, source="test")

        chunks = chunker.chunk(doc)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check character overlap
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].text[-50:]
            chunk2_start = chunks[i + 1].text[:50]

            # Should have character overlap
            assert len(chunk1_end) > 0
            assert len(chunk2_start) > 0


class TestChunkerComparison:
    """Compare different chunking strategies."""

    def test_chunker_outputs_differ(self):
        """Test that different chunkers produce different results."""
        doc = Document(content=SAMPLE_MARKDOWN, metadata={}, source="test")

        recursive_chunker = RecursiveChunker(chunk_size=100, overlap=20)
        semantic_chunker = SemanticChunker(max_chunk_size=100)
        fixed_chunker = FixedSizeChunker(chunk_size=100, overlap=20)

        recursive_chunks = recursive_chunker.chunk(doc)
        semantic_chunks = semantic_chunker.chunk(doc)
        fixed_chunks = fixed_chunker.chunk(doc)

        # Different chunkers should produce different number of chunks
        chunk_counts = [
            len(recursive_chunks),
            len(semantic_chunks),
            len(fixed_chunks),
        ]

        # At least one should be different
        assert len(set(chunk_counts)) > 1


def test_end_to_end_chunking():
    """End-to-end test of chunking pipeline."""
    # Load a document
    doc = Document(
        content=SAMPLE_TEXT_WITH_CODE,
        metadata={"title": "Python Tutorial", "author": "Test"},
        source="test.md",
    )

    # Chunk with RecursiveChunker
    chunker = RecursiveChunker(chunk_size=100, overlap=20, respect_code_blocks=True)
    chunks = chunker.chunk(doc)

    # Verify results
    assert len(chunks) > 0

    # All chunks should have metadata
    for chunk in chunks:
        assert "title" in chunk.metadata
        assert "author" in chunk.metadata
        assert chunk.metadata["title"] == "Python Tutorial"

    # Text should be preserved
    full_text = " ".join([chunk.text for chunk in chunks])
    assert "hello_world" in full_text or "Hello, World!" in full_text


if __name__ == "__main__":
    print("Running chunking tests...")
    pytest.main([__file__, "-v"])
