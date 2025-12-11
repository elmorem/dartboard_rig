# Chunking Strategies

## Overview

**Chunking** is the process of splitting long documents into smaller, semantically coherent passages that can be independently embedded and retrieved. Effective chunking is critical for RAG systems because:

1. **Embedding models have length limits** (typically 512 tokens)
2. **Retrieval accuracy improves** with focused, topic-specific chunks
3. **LLM context windows are finite** - only relevant chunks should be included

## Current Implementation Status

✅ **IMPLEMENTED**: As of December 11, 2025, the chunking module is **fully implemented** and tested.

**Implementation Status**:
- ✅ Document loaders (PDF, Markdown, Code) - Complete
- ✅ Chunking pipeline - **COMPLETE** (SentenceChunker, EmbeddingSemanticChunker, SemanticChunker, FixedSizeChunker)
- ✅ Comprehensive test suite (31 tests passing)
- ✅ End-to-end integration pipeline
- 📋 See implementation at [`dartboard/ingestion/chunking.py`](../dartboard/ingestion/chunking.py)

## Recommended Chunking Strategies

### 1. Fixed-Size Chunking (Simple, Baseline)

Split documents into fixed-size chunks with overlap:

```python
def fixed_size_chunker(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping fixed-size chunks.

    Args:
        text: Input text
        chunk_size: Target chunk size in tokens
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    tokens = text.split()
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk_tokens))

    return chunks
```

**Pros**:
- Simple to implement
- Consistent chunk sizes
- Fast processing

**Cons**:
- May split mid-sentence or mid-paragraph
- No semantic awareness
- Can break context across chunks

**Best For**: Quick prototyping, uniform documents

### 2. Sentence-Aware Chunking (Recommended)

Split on sentence boundaries while respecting chunk size limits:

```python
import nltk
nltk.download('punkt')

def sentence_chunker(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk text respecting sentence boundaries.

    Args:
        text: Input text
        chunk_size: Target chunk size in tokens
        overlap: Overlap in tokens

    Returns:
        List of chunks
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap (last N tokens)
            if overlap > 0:
                overlap_text = " ".join(current_chunk).split()[-overlap:]
                current_chunk = [" ".join(overlap_text)]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**Pros**:
- Preserves sentence integrity
- More semantically coherent
- Better for question answering

**Cons**:
- Variable chunk sizes
- Slightly more complex

**Best For**: General-purpose RAG systems (recommended default)

### 3. Recursive Chunking (Hierarchical)

Split on natural boundaries (paragraphs → sentences → words):

```python
def recursive_chunker(
    text: str,
    chunk_size: int = 512,
    separators: List[str] = ["\n\n", "\n", ". ", " "]
) -> List[str]:
    """
    Recursively split text on natural boundaries.

    Args:
        text: Input text
        chunk_size: Target size in tokens
        separators: List of separators to try (in order)

    Returns:
        List of chunks
    """
    chunks = []

    def split_recursive(text: str, sep_index: int = 0):
        tokens = text.split()
        if len(tokens) <= chunk_size:
            chunks.append(text)
            return

        if sep_index >= len(separators):
            # Fallback to word-level splitting
            for i in range(0, len(tokens), chunk_size):
                chunks.append(" ".join(tokens[i:i+chunk_size]))
            return

        # Try current separator
        separator = separators[sep_index]
        parts = text.split(separator)

        current_part = []
        for part in parts:
            part_tokens = part.split()
            if len(" ".join(current_part + [part]).split()) > chunk_size:
                if current_part:
                    split_recursive(separator.join(current_part), sep_index + 1)
                current_part = [part]
            else:
                current_part.append(part)

        if current_part:
            split_recursive(separator.join(current_part), sep_index + 1)

    split_recursive(text)
    return chunks
```

**Pros**:
- Respects document structure (paragraphs, sections)
- Most semantically coherent
- Adapts to content

**Cons**:
- More complex implementation
- Variable chunk sizes
- Slower processing

**Best For**: Structured documents (articles, papers, documentation)

### 4. Semantic Chunking (Advanced)

Split based on semantic similarity of adjacent sentences:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunker(
    text: str,
    embedding_model: SentenceTransformer,
    similarity_threshold: float = 0.75,
    max_chunk_size: int = 512
) -> List[str]:
    """
    Chunk based on semantic similarity between sentences.

    Args:
        text: Input text
        embedding_model: Model for sentence embeddings
        similarity_threshold: Min similarity to stay in same chunk
        max_chunk_size: Maximum chunk size in tokens

    Returns:
        Semantically coherent chunks
    """
    sentences = nltk.sent_tokenize(text)
    embeddings = embedding_model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_length = len(sentences[0].split())

    for i in range(1, len(sentences)):
        # Compute similarity to previous sentence
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )

        sentence_length = len(sentences[i].split())

        # Start new chunk if dissimilar or too long
        if similarity < similarity_threshold or current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_length = sentence_length
        else:
            current_chunk.append(sentences[i])
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

**Pros**:
- Highest semantic coherence
- Adaptive to content transitions
- Best for complex documents

**Cons**:
- Slowest (requires embeddings)
- Most complex
- Requires embedding model

**Best For**: Research papers, technical documentation, multi-topic articles

## Chunking Parameters

### Chunk Size

Recommended sizes based on use case:

| Use Case | Chunk Size (tokens) | Reasoning |
|----------|---------------------|-----------|
| Short Q&A | 128-256 | Focused answers, fast retrieval |
| **General RAG** | **384-512** | Balances context and specificity |
| Long-form content | 512-1024 | Preserves more context |
| Code documentation | 256-512 | Function/class-level granularity |

**Default Recommendation**: 512 tokens (matches embedding model limits)

### Overlap

Overlap prevents context loss at chunk boundaries:

| Overlap (tokens) | Trade-off |
|------------------|-----------|
| 0 | No redundancy, risk of losing context |
| 25-50 | Minimal redundancy, some context preservation |
| **50-100** | **Good balance (recommended)** |
| 100-200 | High redundancy, ensures context |

**Default Recommendation**: 50 tokens (~10% of chunk size)

### Chunk Size Calculation

```python
def estimate_tokens(text: str) -> int:
    """Rough token estimate (actual tokenization is model-specific)."""
    # Rule of thumb: 1 token ≈ 4 characters for English
    return len(text) // 4

# Or use actual tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokens = tokenizer.tokenize(text)
return len(tokens)
```

## Implementation

### Module Structure

```
dartboard/ingestion/chunking.py
├── TokenCounter (tiktoken-based token counting)
├── Document (data model)
├── Chunk (data model)
├── RecursiveChunker (sentence-aware with overlap)
├── SentenceChunker (alias for RecursiveChunker - RECOMMENDED DEFAULT)
├── EmbeddingSemanticChunker (embedding-based semantic chunking - NEW!)
├── SemanticChunker (paragraph/section-based)
└── FixedSizeChunker (simple token-based)
```

### Actual API

```python
from dartboard.ingestion.chunking import SentenceChunker, Document
from dartboard.ingestion.loaders import PDFLoader
from dartboard.embeddings import SentenceTransformerModel

# Load document
loader = PDFLoader()
documents = loader.load("whitepaper.pdf")

# Chunk documents with SentenceChunker (RECOMMENDED)
chunker = SentenceChunker(
    chunk_size=512,
    overlap=50,
    respect_code_blocks=True  # Preserves code blocks
)

chunks = []
for doc in documents:
    doc_chunks = chunker.chunk(doc)  # Returns List[Chunk]
    chunks.extend(doc_chunks)

# Each chunk has:
# - chunk.text: Full text content
# - chunk.metadata: Dict with source, chunk_index, chunk_size
# - chunk.chunk_index: Sequential index
# - chunk.token_count: Token count (optional)

# Embed chunks
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
embeddings = embedding_model.encode([c.text for c in chunks])

# Add to vector store
vector_store.add(chunks)
```

### Using EmbeddingSemanticChunker (Advanced)

```python
from dartboard.ingestion.chunking import EmbeddingSemanticChunker
from dartboard.embeddings import SentenceTransformerModel

# Load embedding model
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")

# Create semantic chunker
chunker = EmbeddingSemanticChunker(
    embedding_model=embedding_model,
    similarity_threshold=0.75,  # Lower = fewer chunks
    max_chunk_size=512
)

# Chunk based on semantic similarity
chunks = chunker.chunk(document)

# Chunks are grouped by semantic similarity
# Chunks have metadata["semantic_coherence"] = True
```

## Best Practices

### 1. Choose Appropriate Chunk Size

```python
# For embedding model with 512 token limit
chunker = SentenceChunker(chunk_size=512, overlap=50)

# Leave headroom for query tokens in context window
effective_chunk_size = 512 - 50  # Reserve 50 tokens for query
```

### 2. Add Metadata for Provenance

```python
chunk.metadata = {
    "source_file": "document.pdf",
    "page_number": 5,
    "chunk_index": 12,
    "total_chunks": 50,
    "section_title": "Introduction to ML",
    "document_title": "Machine Learning Basics"
}
```

### 3. Preserve Context with Overlap

```python
# Overlapping chunks share context
chunk1 = "...machine learning is a subset of AI. It enables..."
chunk2 = "...It enables computers to learn from data..."  # Overlap: "It enables"
```

### 4. Handle Edge Cases

```python
# Very short documents
if len(text.split()) < chunk_size:
    return [text]  # Single chunk

# Very long sentences
if sentence_length > chunk_size:
    # Split long sentence mid-way
    return split_long_sentence(sentence, chunk_size)
```

## Document Type-Specific Strategies

### PDFs (Research Papers)

```python
# Respect sections, preserve equations, handle figures
chunker = RecursiveChunker(
    chunk_size=512,
    separators=["\n\n", "\n", ". "],  # Paragraph → sentence
    preserve_sections=True
)
```

### Markdown (Documentation)

```python
# Split on headers
chunker = RecursiveChunker(
    chunk_size=512,
    separators=["## ", "### ", "\n\n", "\n"],  # Header-aware
)
```

### Code Repositories

```python
# Function/class-level chunking
chunker = CodeChunker(
    language="python",
    chunk_by="function",  # or "class", "file"
    max_lines=100
)
```

## Evaluation Metrics

### Chunk Quality Metrics

1. **Average Chunk Size**: Should be close to target
2. **Chunk Size Variance**: Lower is better (consistent sizes)
3. **Semantic Coherence**: Measure within-chunk similarity
4. **Context Preservation**: Test on split sentences/paragraphs

### Measuring Chunking Impact on Retrieval

```python
# Compare retrieval metrics with different chunking strategies
for chunker in [FixedSizeChunker(), SentenceChunker(), RecursiveChunker()]:
    chunks = chunker.chunk(documents)
    # ... embed and index ...
    results = retriever.retrieve(queries, chunks)
    ndcg = evaluate_ndcg(results, ground_truth)
    print(f"{chunker}: NDCG@10 = {ndcg:.4f}")

# Expected output:
# FixedSizeChunker: NDCG@10 = 0.62
# SentenceChunker: NDCG@10 = 0.71  ✓
# RecursiveChunker: NDCG@10 = 0.74  ✓✓
```

## References

### Recommended Reading

1. **LangChain Text Splitters**: [https://python.langchain.com/docs/modules/data_connection/document_transformers/](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
2. **Chunking Strategies Guide**: [Pinecone - Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
3. **LlamaIndex**: [https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)

### Implementation References

- **Planned Module**: `dartboard/ingestion/chunking.py` (coming in Phase 1)
- **Document Loaders**: [dartboard/ingestion/loaders.py](../dartboard/ingestion/loaders.py) (complete)
- **Roadmap**: [RAG_IMPLEMENTATION_SUMMARY.md](../RAG_IMPLEMENTATION_SUMMARY.md)

## Summary

Effective chunking is essential for RAG performance. **Sentence-aware chunking with 512 tokens and 50 token overlap** is recommended as the default strategy for most use cases.

**Key Takeaways**:

- ✅ Use **SentenceChunker** as default (512 tokens, 50 overlap) - **IMPLEMENTED**
- ✅ Use **EmbeddingSemanticChunker** for highest quality (slower but more coherent) - **IMPLEMENTED**
- ✅ Add rich **metadata** to chunks for provenance tracking - **IMPLEMENTED**
- ✅ Test different strategies and measure **retrieval impact**
- ✅ Match chunk size to **embedding model limits** (typically 512 tokens)
- ✅ **COMPLETE**: All chunkers implemented with comprehensive tests (31 passing)

## Metadata Enrichment (NEW!)

The chunking pipeline now supports automatic metadata enrichment:

**Features:**

- Extract section titles from Markdown headings
- Build hierarchical heading paths (breadcrumbs)
- Track page numbers for PDF documents
- Maintain document structure and hierarchy

**Usage:**

```python
from dartboard.ingestion.metadata_extractors import MetadataEnricher

# Enrich chunks with metadata
enricher = MetadataEnricher()
enriched_metadata = enricher.enrich_markdown_chunk(
    chunk_text=chunk.text,
    chunk_start=position_in_document,
    document_content=full_document,
    base_metadata=chunk.metadata
)

# Result includes:
# - section_title: Current section name
# - heading_path: Full breadcrumb (e.g., "Intro > Background > Details")
# - heading_level: Heading depth (1, 2, 3, etc.)
```

**Demo:**

```bash
python demo_metadata_enrichment.py
```

## Testing

Run chunking tests:

```bash
# All chunking tests
pytest test_chunking.py test_embedding_semantic_chunking.py -v

# Metadata enrichment tests
pytest test_metadata_extraction.py -v

# Integration demo
python demo_chunking_endtoend.py
```

**Test Coverage**:

- 18 tests for RecursiveChunker/SentenceChunker
- 13 tests for EmbeddingSemanticChunker
- 17 tests for metadata enrichment (NEW!)
- Edge cases: empty documents, single sentences, code blocks
- Metadata preservation and token counting
