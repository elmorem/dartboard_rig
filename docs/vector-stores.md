# Vector Stores

## Overview

Vector stores are specialized databases optimized for storing and searching high-dimensional embedding vectors. They enable fast similarity search, which is the foundation of semantic retrieval in RAG systems.

**Key Concepts:**
- **Embeddings**: Dense vector representations of text (e.g., 384 or 768 dimensions)
- **Similarity Search**: Finding vectors closest to a query vector
- **Index**: Data structure for efficient nearest neighbor search
- **Persistence**: Saving and loading vector stores to/from disk

Dartboard supports multiple vector store backends through a common interface.

## Vector Store Interface

All vector stores implement the `VectorStore` abstract base class:

```python
from dartboard.storage.vector_store import VectorStore

class VectorStore(ABC):
    """Abstract base class for vector storage."""

    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the store."""
        pass

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Chunk]:
        """Search for k most similar chunks."""
        pass

    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass

    def count(self) -> int:
        """Return total number of chunks."""
        pass

    def list_all(self) -> List[Chunk]:
        """Return all chunks in the store."""
        pass
```

## FAISS Vector Store

### Overview

FAISS (Facebook AI Similarity Search) is a high-performance library for similarity search. The `FAISSStore` implementation provides in-memory or disk-persisted vector storage.

**Advantages:**
- ✅ Fast similarity search (optimized C++ backend)
- ✅ No external dependencies or API keys
- ✅ Runs locally (CPU or GPU)
- ✅ Optional disk persistence
- ✅ Good for development and small-to-medium datasets

**Disadvantages:**
- ❌ No built-in multi-tenancy
- ❌ Requires manual persistence management
- ❌ Deletion requires index rebuild
- ❌ Memory-constrained (all data in RAM or disk)

### Quick Start

```python
from dartboard.storage.vector_store import FAISSStore
from dartboard.datasets.models import Chunk
import numpy as np

# Create store
store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"  # Optional: auto-save to disk
)

# Add chunks
chunks = [
    Chunk(
        id="chunk1",
        text="Machine learning is a subset of AI.",
        embedding=np.random.randn(384),
        metadata={"source": "ml_textbook.pdf", "page": 1}
    ),
    Chunk(
        id="chunk2",
        text="Deep learning uses neural networks.",
        embedding=np.random.randn(384),
        metadata={"source": "ml_textbook.pdf", "page": 2}
    )
]

store.add(chunks)

# Search
query_embedding = np.random.randn(384)
results = store.search(query_embedding, k=5)

print(f"Found {len(results)} similar chunks")
for chunk in results:
    print(f"- {chunk.text[:50]}...")
```

### Initialization

```python
from dartboard.storage.vector_store import FAISSStore

# In-memory store (no persistence)
store = FAISSStore(embedding_dim=384)

# Persistent store (auto-saves to disk)
store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/my_index"
)
```

**Parameters:**
- `embedding_dim` (int): Dimension of embedding vectors (e.g., 384, 768)
- `persist_path` (str, optional): Path to save/load index files

**Persistence Behavior:**
- If `persist_path` is provided and files exist, index is loaded automatically
- On every `add()` operation, index is saved to disk
- Creates two files: `{path}.index` (FAISS index) and `{path}.meta` (chunk metadata)

### Adding Chunks

```python
# Add single chunk
chunk = Chunk(id="1", text="...", embedding=embedding, metadata={})
store.add([chunk])

# Add batch of chunks
store.add(chunks)

# Embeddings are automatically normalized for cosine similarity
```

**Normalization:**
- All embeddings are L2-normalized before adding to index
- This converts dot product to cosine similarity
- Query embeddings are also normalized during search

### Searching

```python
# Basic search
results = store.search(query_embedding, k=5)

# Search with metadata filters
results = store.search(
    query_embedding,
    k=5,
    filters={"source": "ml_textbook.pdf"}
)

# Search with scores
result = store.similarity_search(
    query_embedding,
    k=5,
    metric="cosine"
)
chunks = result["chunks"]
scores = result["scores"]

for chunk, score in zip(chunks, scores):
    print(f"Score: {score:.4f} - {chunk.text[:50]}")
```

**Parameters:**
- `query_embedding`: Query vector (numpy array)
- `k`: Number of results to return
- `filters`: Optional metadata filters (exact match)
- `metric`: Similarity metric ("cosine" only currently)

**Returns:**
- `search()`: List of Chunk objects
- `similarity_search()`: Dict with "chunks" and "scores" keys

### Metadata Filtering

FAISS doesn't natively support filtering, so `FAISSStore` implements post-retrieval filtering:

```python
# Filter by source
results = store.search(
    query_embedding,
    k=10,
    filters={"source": "research_papers.pdf"}
)

# Filter by multiple fields
results = store.search(
    query_embedding,
    k=10,
    filters={
        "source": "manual.pdf",
        "section": "installation"
    }
)
```

**Note:** Filters use exact match. FAISS retrieves `k` results first, then filters. If many results are filtered out, you may get fewer than `k` results.

### Deleting Chunks

```python
# Delete by IDs
store.delete(["chunk1", "chunk2", "chunk3"])

# This rebuilds the entire index (expensive operation)
```

**Warning:** Deletion rebuilds the entire FAISS index from scratch. For large indexes, this can be slow. Consider:
- Batching deletions
- Creating a new index instead
- Using a database-backed vector store for frequent deletions

### Persistence

```python
# Auto-persistence (on every add)
store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"
)

store.add(chunks)  # Automatically saved to disk

# Manual persistence
store._save()  # Force save

# Loading from disk
store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"  # Auto-loads if files exist
)

print(f"Loaded {store.count()} chunks from disk")
```

**File Structure:**
```
data/
  vector_store.index   # FAISS index (binary)
  vector_store.meta    # Chunk metadata (pickle)
```

### GPU Acceleration

FAISS supports GPU-accelerated search:

```python
import faiss

# After creating store
store = FAISSStore(embedding_dim=384)

# Move index to GPU (requires faiss-gpu)
res = faiss.StandardGpuResources()
store.index = faiss.index_cpu_to_gpu(res, 0, store.index)

# Search is now GPU-accelerated (3-10x faster)
results = store.search(query_embedding, k=5)
```

**Requirements:**
```bash
pip install faiss-gpu  # Instead of faiss-cpu
```

### Advanced: Index Types

The default `FAISSStore` uses `IndexFlatIP` (flat index with inner product). For large datasets, consider advanced index types:

```python
import faiss

# Create quantized index (less memory, slight quality loss)
quantizer = faiss.IndexFlatIP(384)
index = faiss.IndexIVFFlat(quantizer, 384, 100)  # 100 clusters

# Train index (required for IVF indexes)
index.train(training_embeddings)

# Replace store's index
store.index = index
```

**FAISS Index Types:**
- `IndexFlatIP`: Exact search, slow for large datasets
- `IndexIVFFlat`: Faster search with clustering, requires training
- `IndexHNSW`: Graph-based, fast approximate search
- See [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) for more options

## Pinecone Vector Store

### Overview

Pinecone is a managed cloud vector database with built-in scaling and performance optimization.

**Advantages:**
- ✅ Fully managed (no infrastructure)
- ✅ Automatic scaling
- ✅ Built-in metadata filtering
- ✅ Multi-tenancy support (namespaces)
- ✅ Efficient updates and deletes
- ✅ Hybrid search support

**Disadvantages:**
- ❌ Requires API key and internet connection
- ❌ Usage-based pricing
- ❌ Vendor lock-in
- ❌ Network latency

### Quick Start

```python
from dartboard.storage.vector_store import PineconeStore

# Create store
store = PineconeStore(
    api_key="your-api-key",
    environment="us-east-1-aws",
    index_name="dartboard-index",
    namespace="prod"  # Optional: for multi-tenancy
)

# Use same interface as FAISS
store.add(chunks)
results = store.search(query_embedding, k=5)
```

### Initialization

```python
from dartboard.storage.vector_store import PineconeStore

store = PineconeStore(
    api_key="pk-...",           # Pinecone API key
    environment="us-east-1-aws", # Pinecone environment
    index_name="my-index",       # Index name (must exist)
    namespace="production"       # Optional namespace
)
```

**Parameters:**
- `api_key`: Pinecone API key
- `environment`: Pinecone environment (e.g., "us-east-1-aws")
- `index_name`: Name of Pinecone index (must be created beforehand)
- `namespace`: Optional namespace for multi-tenancy

**Setup Requirements:**
```bash
pip install pinecone-client
```

**Creating Pinecone Index:**
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create index (one-time setup)
pc.create_index(
    name="dartboard-index",
    dimension=384,  # Match your embedding model
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### Adding Chunks

```python
# Pinecone automatically batches uploads
store.add(chunks)

# Chunks are cached locally and upserted to Pinecone
```

**Behavior:**
- Chunks are stored in Pinecone with metadata
- Text is truncated to 1000 chars for Pinecone metadata limits
- Full chunks are cached locally for fast retrieval

### Searching

```python
# Search with native metadata filtering
results = store.search(
    query_embedding,
    k=5,
    filters={"source": "manual.pdf"}
)

# Pinecone uses native filtering (faster than post-filtering)
```

**Metadata Filtering:**
Pinecone supports rich filtering:
```python
# Exact match
filters = {"source": "paper.pdf"}

# Multiple conditions (automatic $and)
filters = {"source": "paper.pdf", "section": "methods"}
```

### Deleting Chunks

```python
# Efficient deletion (no index rebuild)
store.delete(["chunk1", "chunk2"])

# Much faster than FAISS deletion
```

### Namespaces (Multi-Tenancy)

```python
# Production namespace
prod_store = PineconeStore(
    api_key=api_key,
    environment=env,
    index_name="shared-index",
    namespace="production"
)

# Development namespace (same index, isolated data)
dev_store = PineconeStore(
    api_key=api_key,
    environment=env,
    index_name="shared-index",
    namespace="development"
)

# Completely isolated
prod_store.add(prod_chunks)
dev_store.add(dev_chunks)

prod_results = prod_store.search(query, k=5)  # Only searches production
```

## Choosing a Vector Store

### Decision Matrix

| Feature | FAISS | Pinecone |
|---------|-------|----------|
| **Setup Complexity** | ⭐ Simple | ⭐⭐ Moderate |
| **Cost** | Free | Usage-based |
| **Performance** | ⭐⭐⭐ Fast (local) | ⭐⭐ Fast (network latency) |
| **Scalability** | ⭐⭐ Memory-limited | ⭐⭐⭐ Auto-scaling |
| **Metadata Filtering** | ⭐⭐ Post-filtering | ⭐⭐⭐ Native filtering |
| **Updates/Deletes** | ⭐ Slow (rebuild) | ⭐⭐⭐ Fast |
| **Multi-Tenancy** | ❌ | ✅ Namespaces |
| **Persistence** | ⭐⭐ Manual | ⭐⭐⭐ Automatic |
| **Best For** | Development, small-medium datasets, local | Production, large datasets, multi-user |

### Use FAISS When:

- ✅ Developing or prototyping
- ✅ Dataset fits in memory (<10M vectors)
- ✅ Running locally or on single server
- ✅ Want zero external dependencies
- ✅ Cost is a concern
- ✅ Data privacy requires local storage

**Example Use Cases:**
- Research projects
- Personal knowledge bases
- Local RAG applications
- CI/CD testing
- Offline applications

### Use Pinecone When:

- ✅ Production deployment
- ✅ Large or growing dataset
- ✅ Need multi-tenancy
- ✅ Frequent updates and deletes
- ✅ Want managed infrastructure
- ✅ Need distributed architecture

**Example Use Cases:**
- SaaS applications
- Multi-tenant RAG platforms
- Large-scale document search
- Customer-facing products
- Microservices architecture

## Performance Benchmarks

### FAISS Performance

**Search Latency (1M vectors, 384-dim, CPU):**
- `k=5`: ~5ms
- `k=10`: ~8ms
- `k=50`: ~15ms

**Search Latency (GPU acceleration):**
- 3-10x faster than CPU
- `k=5`: ~0.5-2ms

**Insertion:**
- Batch of 1000: ~50ms (CPU) / ~10ms (GPU)
- Includes normalization and index update

**Memory:**
- ~1.5KB per vector (384-dim)
- 1M vectors ≈ 1.5GB RAM

### Pinecone Performance

**Search Latency:**
- Network latency: 20-100ms (depending on region)
- Search time: 5-20ms
- Total: 25-120ms

**Insertion:**
- Batch of 1000: ~200-500ms
- Async upsert for better throughput

**Scaling:**
- Automatically handles millions to billions of vectors
- No memory constraints

### Optimization Tips

**FAISS:**
```python
# 1. Use GPU
store.index = faiss.index_cpu_to_gpu(res, 0, store.index)

# 2. Use batch operations
store.add(large_chunk_list)  # Better than multiple small adds

# 3. Use advanced index for large datasets
index = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index.train(embeddings)
```

**Pinecone:**
```python
# 1. Use namespaces for isolation
store = PineconeStore(..., namespace="user123")

# 2. Batch operations
store.add(chunks)  # Automatically batched

# 3. Use pod-based indexes for higher performance (vs serverless)
```

## Migration Between Vector Stores

### FAISS to Pinecone

```python
from dartboard.storage.vector_store import FAISSStore, PineconeStore

# 1. Load from FAISS
faiss_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"
)

# 2. Get all chunks
chunks = faiss_store.list_all()

# 3. Create Pinecone store
pinecone_store = PineconeStore(
    api_key=api_key,
    environment=env,
    index_name="dartboard-index"
)

# 4. Migrate data
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    pinecone_store.add(batch)
    print(f"Migrated {i+batch_size}/{len(chunks)} chunks")

print(f"✓ Migration complete: {pinecone_store.count()} chunks")
```

### Pinecone to FAISS

```python
# 1. Load from Pinecone
pinecone_store = PineconeStore(...)

# 2. Get all chunks (from cache)
chunks = pinecone_store.list_all()

# Note: This only includes chunks added in current session
# For full export, use Pinecone API directly

# 3. Create FAISS store
faiss_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/vector_store"
)

# 4. Add chunks
faiss_store.add(chunks)
```

## Integration Examples

### With Ingestion Pipeline

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import PDFLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Create vector store
vector_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/papers"
)

# Create pipeline
pipeline = create_pipeline(
    loader=PDFLoader(),
    embedding_model=SentenceTransformerModel("all-MiniLM-L6-v2"),
    vector_store=vector_store
)

# Ingest documents
pipeline.ingest_batch([
    "paper1.pdf",
    "paper2.pdf",
    "paper3.pdf"
])

print(f"Vector store size: {vector_store.count()} chunks")
```

### With Retrieval

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Load vector store
vector_store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/papers"
)

# Create retriever
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

# Retrieve
results = retriever.retrieve("What is attention mechanism?", k=5)

for chunk in results:
    print(f"- {chunk.text[:100]}...")
```

### With API

```python
from fastapi import FastAPI, Depends
from dartboard.storage.vector_store import FAISSStore

app = FastAPI()

# Singleton vector store
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSStore(
            embedding_dim=384,
            persist_path="./data/vector_store"
        )
    return _vector_store

@app.get("/stats")
async def get_stats(store = Depends(get_vector_store)):
    return {
        "total_chunks": store.count(),
        "type": "faiss"
    }
```

## Best Practices

### 1. Always Match Embedding Dimensions

```python
from dartboard.config import get_embedding_config

# Get config
config = get_embedding_config()

# Use consistent dimension everywhere
vector_store = FAISSStore(embedding_dim=config.embedding_dim)
embedding_model = SentenceTransformerModel(config.model_name)
```

### 2. Use Persistence in Production

```python
# ✅ Good: Persistent store
store = FAISSStore(
    embedding_dim=384,
    persist_path="./data/production_index"
)

# ❌ Bad: In-memory only (lost on restart)
store = FAISSStore(embedding_dim=384)
```

### 3. Validate Store Before Use

```python
store = FAISSStore(embedding_dim=384, persist_path="./data/store")

# Check if populated
if store.count() == 0:
    print("Warning: Vector store is empty!")
    # Run ingestion...
```

### 4. Handle Dimension Mismatches

```python
try:
    # This will fail if dimensions don't match
    store.search(wrong_dim_embedding, k=5)
except Exception as e:
    print(f"Dimension mismatch: {e}")
    # Re-create store with correct dimensions
```

### 5. Use Metadata for Organization

```python
chunks = [
    Chunk(
        id="1",
        text="...",
        embedding=emb,
        metadata={
            "source": "manual.pdf",
            "section": "installation",
            "page": 5,
            "date": "2024-01-01"
        }
    )
]

# Later: filter by metadata
results = store.search(
    query,
    k=10,
    filters={"section": "installation"}
)
```

## Troubleshooting

### Issue: FAISS import error

```python
ImportError: faiss-cpu is required for FAISSStore
```

**Solution:**
```bash
pip install faiss-cpu  # For CPU
# or
pip install faiss-gpu  # For GPU
```

### Issue: Dimension mismatch

```python
ValueError: Embedding dimension mismatch: expected 384, got 768
```

**Solution:**
```python
# Ensure vector store matches embedding model
config = get_embedding_config()
store = FAISSStore(embedding_dim=config.embedding_dim)
```

### Issue: Persistence files not found

```python
# Store created but no files on disk
store = FAISSStore(embedding_dim=384, persist_path="./data/store")
# Files not created until first add()
```

**Solution:**
```python
# Files only created after adding data
store.add(chunks)  # Now ./data/store.index and ./data/store.meta exist
```

### Issue: Pinecone connection timeout

```python
PineconeException: Connection timeout
```

**Solution:**
```python
# 1. Check API key
# 2. Verify environment name
# 3. Check internet connection
# 4. Verify index exists

from pinecone import Pinecone
pc = Pinecone(api_key=api_key)
print(pc.list_indexes())  # Should include your index
```

### Issue: Search returns fewer than k results

**Cause:** Metadata filtering excluded some results (FAISS only)

**Solution:**
```python
# Request more results before filtering
results = store.search(
    query,
    k=20,  # Higher k to account for filtering
    filters={"source": "rare_document.pdf"}
)
```

## See Also

- [Ingestion Pipeline](./ingestion-pipeline.md) - How to populate vector stores
- [Dense Retrieval](./dense-retrieval.md) - Using vector stores for retrieval
- [Retrieval Methods](./retrieval-methods.md) - All retrieval approaches
- [Configuring Embedding Models](./configuring-embedding-models.md) - Embedding configuration
