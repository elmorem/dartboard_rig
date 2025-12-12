# Document Ingestion Pipeline

## Overview

The Dartboard ingestion pipeline transforms raw documents into searchable, embedded chunks stored in a vector database. It orchestrates the complete workflow from document loading through storage.

**Pipeline Flow:**
```
Load Documents → Chunk Text → Generate Embeddings → Store in Vector DB
```

**Key Components:**
- **Document Loaders**: Load PDFs, Markdown, code repositories
- **Chunkers**: Split documents into semantic passages
- **Embedding Models**: Convert text to vector representations
- **Vector Store**: Persist embeddings for retrieval

## Quick Start

### Basic Ingestion

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import PDFLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore
from dartboard.config import get_embedding_config

# Setup components
config = get_embedding_config()
loader = PDFLoader()
embedding_model = SentenceTransformerModel(config.model_name)
vector_store = FAISSStore(embedding_dim=config.embedding_dim)

# Create pipeline
pipeline = create_pipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunk_size=512,
    overlap=50
)

# Ingest document
result = pipeline.ingest("path/to/document.pdf", track_progress=True)

print(f"Processed {result.documents_processed} documents")
print(f"Created {result.chunks_created} chunks")
print(f"Stored {result.chunks_stored} chunks")
```

### Batch Ingestion

```python
# Ingest multiple documents
sources = [
    "documents/paper1.pdf",
    "documents/paper2.pdf",
    "documents/readme.md"
]

results = pipeline.ingest_batch(sources, track_progress=True)

# Check results
for i, result in enumerate(results):
    if result.status == "success":
        print(f"✓ {sources[i]}: {result.chunks_created} chunks")
    else:
        print(f"✗ {sources[i]}: {result.errors}")
```

## Document Loaders

### Overview

Document loaders extract text content from various file formats and return `Document` objects with metadata.

**Available Loaders:**
- `PDFLoader` - Extract text from PDF files
- `MarkdownLoader` - Load Markdown with frontmatter
- `CodeRepositoryLoader` - Load code files from repositories
- `DirectoryLoader` - Auto-detect and load all supported files

### Document Object

```python
from dartboard.ingestion.loaders import Document

doc = Document(
    content="Full text content...",
    metadata={
        "source_type": "pdf",
        "file_name": "paper.pdf",
        "title": "Research Paper",
        "author": "John Doe",
        "num_pages": 10
    },
    source="/path/to/paper.pdf"
)
```

### PDFLoader

Extract text from PDF documents with metadata extraction.

```python
from dartboard.ingestion.loaders import PDFLoader

loader = PDFLoader(extract_images=False)
documents = loader.load("research_paper.pdf")

doc = documents[0]
print(f"Pages: {doc.metadata['num_pages']}")
print(f"Title: {doc.metadata.get('title', 'Unknown')}")
print(f"Content length: {len(doc.content)} chars")
```

**Features:**
- Extracts text from all pages
- Preserves PDF metadata (title, author, creator, subject)
- Handles multi-page documents
- Returns single Document with full text

**Dependencies:**
```bash
pip install pypdf
```

**Metadata Fields:**
- `source_type`: "pdf"
- `file_name`: Filename
- `file_path`: Absolute path
- `num_pages`: Page count
- `title`, `author`, `creator`, `subject`: PDF metadata (if available)

### MarkdownLoader

Load Markdown files with optional frontmatter and code block extraction.

```python
from dartboard.ingestion.loaders import MarkdownLoader

loader = MarkdownLoader(extract_code_blocks=True)
documents = loader.load("README.md")

doc = documents[0]
print(f"Title: {doc.metadata.get('title', 'No title')}")
print(f"Code blocks: {len(doc.metadata.get('code_blocks', []))}")
```

**Features:**
- Extracts YAML frontmatter
- Optionally extracts code blocks separately
- Preserves Markdown formatting
- Returns Document with enriched metadata

**Dependencies:**
```bash
pip install pyyaml  # Optional, for frontmatter
```

**Metadata Fields:**
- `source_type`: "markdown"
- `file_name`: Filename
- `file_path`: Absolute path
- Frontmatter fields (if present)
- `code_blocks`: List of `{language, code}` dicts (if enabled)

**Example with Frontmatter:**
```markdown
---
title: "My Document"
author: "Jane Smith"
date: "2024-01-01"
---

# Content here...
```

### CodeRepositoryLoader

Load code files from a repository with language detection.

```python
from dartboard.ingestion.loaders import CodeRepositoryLoader

loader = CodeRepositoryLoader(
    file_extensions=[".py", ".js", ".ts", ".md"],
    exclude_patterns=["node_modules", "__pycache__", ".git", "venv"]
)

documents = loader.load("path/to/repository")

print(f"Loaded {len(documents)} files")
for doc in documents[:3]:
    print(f"  - {doc.metadata['relative_path']} ({doc.metadata['language']})")
```

**Features:**
- Recursively walks directory tree
- Filters by file extension
- Excludes common build/cache directories
- Auto-detects programming language
- Returns one Document per file

**Default Extensions:**
`.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.rs`, `.go`, `.rb`, `.php`, `.md`

**Default Exclusions:**
`node_modules`, `__pycache__`, `.git`, `venv`, `.venv`, `build`, `dist`, `.pytest_cache`, `.mypy_cache`

**Metadata Fields:**
- `source_type`: "code"
- `file_name`: Filename
- `file_path`: Absolute path
- `relative_path`: Path relative to repository root
- `language`: Detected programming language
- `extension`: File extension
- `repository`: Repository root path

### DirectoryLoader

Automatically detect and load all supported files from a directory.

```python
from dartboard.ingestion.loaders import DirectoryLoader

loader = DirectoryLoader()
documents = loader.load("documents/")

print(f"Loaded {len(documents)} documents")
```

**Features:**
- Auto-detects file types
- Uses appropriate loader for each file
- Recursively processes subdirectories
- Skips unsupported file types

**Supported Types:**
- `.pdf` → PDFLoader
- `.md` → MarkdownLoader

## Ingestion Pipeline

### IngestionPipeline

Orchestrates the complete ingestion workflow.

```python
from dartboard.ingestion.pipeline import IngestionPipeline
from dartboard.ingestion.loaders import PDFLoader
from dartboard.ingestion.chunking import RecursiveChunker
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Create components
loader = PDFLoader()
chunker = RecursiveChunker(chunk_size=512, overlap=50)
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(embedding_dim=384)

# Create pipeline
pipeline = IngestionPipeline(
    loader=loader,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    batch_size=32  # Embedding batch size
)

# Ingest single document
result = pipeline.ingest("document.pdf", track_progress=True)

# Ingest multiple documents
results = pipeline.ingest_batch(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    track_progress=True
)
```

### Pipeline Workflow

**Step 1: Load Documents**
```python
documents = loader.load(source)
# Returns: List[Document]
```

**Step 2: Chunk Documents**
```python
chunks = chunker.chunk_batch(documents)
# Returns: List[Chunk] with text and metadata
```

**Step 3: Generate Embeddings**
```python
texts = [chunk.text for chunk in chunks]
embeddings = embedding_model.encode(texts, batch_size=32)
# Returns: np.ndarray of shape (n_chunks, embedding_dim)
```

**Step 4: Store in Vector Database**
```python
storage_chunks = [
    Chunk(id=..., text=..., embedding=..., metadata=...)
    for ...
]
vector_store.add(storage_chunks)
```

### IngestionResult

Pipeline returns detailed results:

```python
@dataclass
class IngestionResult:
    documents_processed: int      # Number of documents loaded
    chunks_created: int            # Number of chunks generated
    chunks_stored: int             # Number of chunks stored
    status: str                    # "success" or "failed"
    errors: List[str]              # Error messages (if any)
    metadata: Dict[str, Any]       # Additional metadata
```

**Example:**
```python
result = pipeline.ingest("paper.pdf")

if result.status == "success":
    print(f"✓ Processed {result.documents_processed} documents")
    print(f"✓ Created {result.chunks_created} chunks")
    print(f"✓ Average chunk size: {result.metadata['avg_chunk_size']:.0f} chars")
else:
    print(f"✗ Ingestion failed: {result.errors}")
```

### BatchIngestionPipeline

Pipeline with retry logic for robust batch processing.

```python
from dartboard.ingestion.pipeline import BatchIngestionPipeline

pipeline = BatchIngestionPipeline(
    loader=loader,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    batch_size=32,
    max_retries=3  # Retry failed operations
)

# Ingest with automatic retry
result = pipeline.ingest_with_retry("document.pdf", track_progress=True)
```

**Features:**
- Automatic retry on failure
- Configurable max retries
- Tracks retry attempts in metadata
- Returns result after all retries exhausted

## Factory Functions

### create_pipeline

Convenient factory function to create a standard pipeline.

```python
from dartboard.ingestion import create_pipeline

pipeline = create_pipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunk_size=512,        # Tokens per chunk
    overlap=50,            # Token overlap between chunks
    batch_size=32          # Embedding batch size
)
```

**Creates:**
- `RecursiveChunker` with specified parameters
- `IngestionPipeline` with all components wired together

## Complete Examples

### Example 1: Ingest PDF Research Papers

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import PDFLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore
from dartboard.config import get_embedding_config

# Configuration
config = get_embedding_config()

# Setup
loader = PDFLoader()
embedding_model = SentenceTransformerModel(
    model_name=config.model_name,
    device=config.device
)
vector_store = FAISSStore(
    embedding_dim=config.embedding_dim,
    persist_path="./data/vector_store"
)

# Create pipeline
pipeline = create_pipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunk_size=512,
    overlap=50
)

# Ingest papers
papers = [
    "papers/transformer_attention.pdf",
    "papers/bert_pretraining.pdf",
    "papers/gpt3_language_models.pdf"
]

results = pipeline.ingest_batch(papers, track_progress=True)

# Summary
total_chunks = sum(r.chunks_created for r in results)
print(f"\n✓ Ingested {len(papers)} papers")
print(f"✓ Created {total_chunks} searchable chunks")
print(f"✓ Vector store size: {vector_store.count()}")
```

### Example 2: Ingest Code Repository

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import CodeRepositoryLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Load Python files only
loader = CodeRepositoryLoader(
    file_extensions=[".py"],
    exclude_patterns=["tests", "__pycache__", ".venv", "build"]
)

# Setup pipeline
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(embedding_dim=384)

pipeline = create_pipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_store=vector_store,
    chunk_size=1024,  # Larger chunks for code
    overlap=100
)

# Ingest repository
result = pipeline.ingest("path/to/my-project", track_progress=True)

print(f"Indexed {result.chunks_created} code chunks")
```

### Example 3: Ingest Mixed Documents

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import DirectoryLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Auto-detect all file types
loader = DirectoryLoader()

# Setup
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/knowledge_base"
)

pipeline = create_pipeline(
    loader=loader,
    embedding_model=embedding_model,
    vector_store=vector_store
)

# Ingest entire directory (PDFs + Markdown)
result = pipeline.ingest("knowledge_base/", track_progress=True)

print(f"Knowledge base ready: {result.chunks_stored} chunks indexed")
```

### Example 4: Custom Pipeline with Metadata Enrichment

```python
from dartboard.ingestion.pipeline import IngestionPipeline
from dartboard.ingestion.loaders import MarkdownLoader
from dartboard.ingestion.chunking import RecursiveChunker
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Custom chunker with semantic chunking
chunker = RecursiveChunker(
    chunk_size=512,
    overlap=50,
    respect_code_blocks=True,      # Keep code blocks intact
    respect_markdown_structure=True # Respect headings
)

# Custom loader
loader = MarkdownLoader(extract_code_blocks=True)

# Setup
embedding_model = SentenceTransformerModel("all-mpnet-base-v2")
vector_store = FAISSStore(embedding_dim=768)

# Custom pipeline
pipeline = IngestionPipeline(
    loader=loader,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    batch_size=16  # Smaller batches for larger model
)

# Ingest with rich metadata
result = pipeline.ingest("documentation.md", track_progress=True)
```

## Configuration

### Chunk Size Selection

**Small Chunks (256-512 tokens):**
- More precise retrieval
- Better for factoid questions
- Higher storage costs
- Use for: Q&A, fact lookup

**Medium Chunks (512-1024 tokens):**
- Balanced precision and context
- Good for most use cases
- Recommended default
- Use for: General RAG, documentation

**Large Chunks (1024-2048 tokens):**
- More context per chunk
- Better for complex reasoning
- Lower retrieval precision
- Use for: Code, long-form content

### Overlap Configuration

**Overlap Purpose:**
- Prevents information loss at chunk boundaries
- Ensures continuous context
- Improves retrieval recall

**Recommended Values:**
- 10-15% of chunk size (e.g., 50 tokens for 512-token chunks)
- Higher overlap for structured content (Markdown, code)
- Lower overlap for unstructured text

### Batch Size

**Embedding Batch Size:**
- Larger batches = faster ingestion
- Limited by GPU memory
- CPU: 32-64 typical
- GPU: 64-256 typical

```python
# CPU (slower, more memory-efficient)
pipeline = create_pipeline(..., batch_size=32)

# GPU (faster, requires more memory)
pipeline = create_pipeline(..., batch_size=128)
```

## Performance Optimization

### Ingestion Speed

**Bottlenecks:**
1. **Embedding generation** (slowest)
2. Document loading
3. Chunking
4. Vector store insertion

**Optimizations:**

```python
# 1. Use GPU for embeddings
embedding_model = SentenceTransformerModel(
    "all-MiniLM-L6-v2",
    device="cuda"  # 3-5x faster
)

# 2. Increase batch size
pipeline = create_pipeline(
    ...,
    batch_size=128  # GPU can handle larger batches
)

# 3. Use faster embedding model
embedding_model = SentenceTransformerModel(
    "all-MiniLM-L6-v2"  # Fast: 14K sentences/sec
    # vs "all-mpnet-base-v2"  # Slow: 3K sentences/sec
)

# 4. Parallelize document loading (future enhancement)
```

### Memory Management

**For Large Document Sets:**

```python
# Process in smaller batches
import glob

pdf_files = glob.glob("documents/*.pdf")
batch_size = 10

for i in range(0, len(pdf_files), batch_size):
    batch = pdf_files[i:i+batch_size]
    results = pipeline.ingest_batch(batch, track_progress=True)
    print(f"Processed batch {i//batch_size + 1}")
```

## Error Handling

### Common Errors

**1. File Not Found**
```python
try:
    result = pipeline.ingest("missing.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

**2. Import Errors (Missing Dependencies)**
```python
# If pypdf not installed
from dartboard.ingestion.loaders import PDFLoader
loader = PDFLoader()  # Raises ImportError with install instructions
```

**3. Embedding Dimension Mismatch**
```python
# Ensure vector store matches embedding model
config = get_embedding_config()
vector_store = FAISSStore(embedding_dim=config.embedding_dim)
```

### Robust Error Handling

```python
from dartboard.ingestion.pipeline import BatchIngestionPipeline

# Pipeline with retry logic
pipeline = BatchIngestionPipeline(
    loader=loader,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    max_retries=3
)

# Ingest with automatic retry
results = []
for source in sources:
    result = pipeline.ingest_with_retry(source, track_progress=True)
    results.append(result)

    if result.status == "failed":
        print(f"Failed after {result.metadata.get('attempts', 0)} attempts: {source}")
```

## Integration with Other Components

### With Retrieval

```python
# 1. Ingest documents
pipeline = create_pipeline(...)
pipeline.ingest_batch(documents)

# 2. Retrieve relevant chunks
from dartboard.retrieval.dense import DenseRetriever

retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

results = retriever.retrieve("What is attention?", k=5)
```

### With Generation

```python
# 1. Ingest
pipeline.ingest_batch(documents)

# 2. Retrieve
retriever = DenseRetriever(embedding_model, vector_store)
chunks = retriever.retrieve(query, k=5)

# 3. Generate
from dartboard.generation import create_generator

generator = create_generator(provider="openai", api_key=api_key)
result = generator.generate(query, chunks)
print(result.answer)
```

### With API

```python
# In FastAPI app
from dartboard.api.dependencies import get_ingestion_pipeline

@app.post("/ingest")
async def ingest_document(
    file: UploadFile,
    pipeline = Depends(get_ingestion_pipeline)
):
    # Save uploaded file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Ingest
    result = pipeline.ingest(temp_path, track_progress=True)

    return {
        "status": result.status,
        "chunks_created": result.chunks_created
    }
```

## Best Practices

### 1. Choose Appropriate Chunk Size

Match chunk size to your use case:
- **Factoid Q&A**: 256-512 tokens
- **Documentation**: 512-1024 tokens
- **Code**: 1024-2048 tokens
- **Long-form reasoning**: 1024+ tokens

### 2. Use Consistent Embedding Models

Always use the same embedding model for ingestion and retrieval:

```python
# Configure once
config = get_embedding_config()

# Use everywhere
embedding_model = SentenceTransformerModel(config.model_name)
```

### 3. Persist Vector Store

Save your vector store to avoid re-ingestion:

```python
vector_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"  # Auto-saves on updates
)
```

### 4. Track Progress for Large Ingestions

```python
# Always use track_progress=True for visibility
result = pipeline.ingest_batch(
    large_document_list,
    track_progress=True  # Shows progress logs
)
```

### 5. Validate Results

```python
result = pipeline.ingest(source)

assert result.status == "success", f"Ingestion failed: {result.errors}"
assert result.chunks_stored > 0, "No chunks were stored"
print(f"✓ Successfully stored {result.chunks_stored} chunks")
```

## Troubleshooting

### Issue: No chunks created

**Possible causes:**
- Document is empty or unreadable
- Chunk size too large for document
- File format not supported

**Solution:**
```python
# Check document content
documents = loader.load(source)
print(f"Document length: {len(documents[0].content)} chars")

# Reduce chunk size
pipeline = create_pipeline(..., chunk_size=256, overlap=25)
```

### Issue: Embedding generation slow

**Solution:**
```python
# 1. Use GPU
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2", device="cuda")

# 2. Use faster model
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")  # vs all-mpnet-base-v2

# 3. Increase batch size
pipeline = create_pipeline(..., batch_size=128)
```

### Issue: Out of memory during ingestion

**Solution:**
```python
# Reduce batch size
pipeline = create_pipeline(..., batch_size=16)

# Or process documents one at a time
for doc in document_list:
    result = pipeline.ingest(doc, track_progress=True)
```

## See Also

- [Chunking Strategies](./chunking-strategies.md) - Chunking algorithms and configuration
- [Configuring Embedding Models](./configuring-embedding-models.md) - Embedding model selection
- [Vector Stores](./vector-stores.md) - Vector database options
- [Metadata Extraction](./metadata-extraction.md) - Enriching chunks with context
- [API Guide](./api-guide.md) - Using ingestion via API
