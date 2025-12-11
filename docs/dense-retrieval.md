# Dense Retrieval

## Overview

Dense retrieval is a **semantic search** method that represents queries and documents as dense vectors (embeddings) in a high-dimensional space. Documents are ranked by vector similarity (typically cosine similarity) to the query vector, enabling retrieval based on meaning rather than exact keyword matches.

Dense retrieval has revolutionized information retrieval by enabling semantic understanding, handling paraphrased queries, and working across languages.

## Core Concept

### From Sparse to Dense

**Traditional (Sparse)**: Documents represented as sparse vectors based on term frequencies
- Dimension = vocabulary size (10K-1M)
- Most values are zero
- Example: "machine learning" → [0, 0, ..., 1, ..., 0, ..., 2, ..., 0]

**Dense Retrieval**: Documents represented as dense embeddings from neural networks
- Dimension = embedding size (384-1024)
- All values are non-zero
- Example: "machine learning" → [0.23, -0.15, 0.67, ..., 0.41]

### Key Advantage

**Semantic Understanding**:
- Query: "how to train ML models"
- Document: "machine learning tutorial"
- Sparse (BM25): Poor match (few overlapping words)
- Dense: Strong match (similar semantic meaning)

## How Dense Retrieval Works

### 1. Embedding Generation

Convert text to fixed-size vectors using sentence transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode query
query = "machine learning applications"
query_embedding = model.encode(query)  # Shape: (384,)

# Encode documents
docs = ["ML is widely used...", "Deep learning..."]
doc_embeddings = model.encode(docs)  # Shape: (2, 384)
```

### 2. Similarity Computation

Calculate cosine similarity between query and document embeddings:

```
similarity(q, d) = (q · d) / (||q|| · ||d||)
```

Where:
- `q · d` = dot product
- `||q||`, `||d||` = L2 norms (vector magnitudes)

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute similarities
scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
```

### 3. Ranking

Sort documents by similarity score (descending) and return top-k:

```python
# Get top-5 most similar documents
top_indices = np.argsort(scores)[::-1][:5]
top_docs = [docs[i] for i in top_indices]
top_scores = [scores[i] for i in top_indices]
```

## Implementation in This Repository

### Architecture

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.storage.vector_store import FAISSVectorStore

# Initialize embedding model
retriever = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    similarity_metric="cosine"  # or "dot", "euclidean"
)

# Vector store with pre-computed embeddings
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.add(chunks_with_embeddings)

# Retrieve
result = retriever.retrieve(
    query="machine learning applications",
    k=10
)
```

### Core Components

#### 1. Embedding Model

Default model: **all-MiniLM-L6-v2**
- Size: 384 dimensions
- Speed: ~14,000 sentences/sec on CPU
- Quality: Good balance of speed and accuracy

```python
retriever = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Other popular models:

```python
# Higher quality, slower
retriever = DenseRetriever(
    model_name="sentence-transformers/all-mpnet-base-v2"  # 768-dim
)

# Multilingual
retriever = DenseRetriever(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Domain-specific (MS MARCO)
retriever = DenseRetriever(
    model_name="sentence-transformers/msmarco-distilbert-base-v4"
)
```

#### 2. Vector Store Integration

Dense retrieval requires efficient similarity search:

```python
from dartboard.storage.vector_store import FAISSVectorStore, PineconeVectorStore

# Option 1: FAISS (local, fast)
faiss_store = FAISSVectorStore(embedding_dim=384)
faiss_store.add(chunks)

# Option 2: Pinecone (cloud, scalable)
pinecone_store = PineconeVectorStore(
    api_key="your-key",
    index_name="my-index",
    embedding_dim=384
)
pinecone_store.add(chunks)

# Use with retriever
retriever = DenseRetriever(vector_store=faiss_store)
result = retriever.retrieve(query, k=10)
```

#### 3. Similarity Metrics

Three similarity metrics supported:

**Cosine Similarity** (default, recommended):
```python
retriever = DenseRetriever(similarity_metric="cosine")
# Range: [-1, 1], higher is more similar
# Normalized: unaffected by vector magnitude
```

**Dot Product**:
```python
retriever = DenseRetriever(similarity_metric="dot")
# Range: (-∞, ∞), higher is more similar
# Fast: no normalization needed
# Best for: pre-normalized embeddings
```

**Euclidean Distance**:
```python
retriever = DenseRetriever(similarity_metric="euclidean")
# Range: [0, ∞), lower is more similar
# Geometric: actual distance in embedding space
```

## Usage Examples

### Basic Retrieval

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.storage.vector_store import FAISSVectorStore
from dartboard.datasets.models import Chunk
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare corpus with embeddings
chunks = []
for doc_id, text in enumerate(documents):
    embedding = model.encode(text)
    chunk = Chunk(
        id=str(doc_id),
        text=text,
        embedding=embedding
    )
    chunks.append(chunk)

# Build vector store
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.add(chunks)

# Initialize retriever
retriever = DenseRetriever(
    vector_store=vector_store,
    embedding_model=model
)

# Query
result = retriever.retrieve("deep learning applications", k=5)

# Print results
for i, chunk in enumerate(result.chunks, 1):
    print(f"{i}. Score: {chunk.score:.4f}")
    print(f"   Text: {chunk.text[:100]}...\n")
```

### Batch Encoding

Efficiently encode multiple documents:

```python
# Encode documents in batches
texts = [chunk.text for chunk in chunks]
embeddings = retriever.encode_batch(texts, batch_size=32)

# Update chunks with embeddings
for chunk, embedding in zip(chunks, embeddings):
    chunk.embedding = embedding

# Add to vector store
vector_store.add(chunks)
```

### GPU Acceleration

Use GPU for faster encoding:

```python
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize on GPU
retriever = DenseRetriever(
    model_name="all-MiniLM-L6-v2",
    vector_store=vector_store
)
retriever.embedding_model.model.to(device)

# ~10x faster encoding on GPU
embeddings = retriever.encode_batch(texts, batch_size=64)
```

### Multi-Query Retrieval

Process multiple queries efficiently:

```python
queries = [
    "machine learning basics",
    "deep neural networks",
    "natural language processing"
]

results = []
for query in queries:
    result = retriever.retrieve(query, k=5)
    results.append(result)

# Analyze results
for query, result in zip(queries, results):
    print(f"\nQuery: {query}")
    print(f"Top result: {result.chunks[0].text[:80]}...")
    print(f"Top score: {result.chunks[0].score:.4f}")
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Encode query | O(L) | L = sequence length, ~5ms on CPU |
| Similarity search (brute force) | O(N · D) | N docs, D dimensions |
| Similarity search (FAISS IVF) | O(√N · D) | With index, sub-linear |
| Similarity search (FAISS HNSW) | O(log N · D) | Approximate, very fast |

### Space Complexity

- **Embeddings**: O(N · D) where N = num docs, D = embedding dim
  - 100K docs × 384 dims × 4 bytes = 153 MB
- **FAISS Index**: Additional ~20-50% overhead for index structure

### Benchmarked Performance (Dec 2025)

Tested on BEIR datasets with all-MiniLM-L6-v2 (384-dim):

| Dataset | Corpus Size | Indexing Time | Query Latency (p95) | Throughput |
|---------|-------------|---------------|---------------------|------------|
| SciFact | 5,183 docs | 2.1s | 45ms | 2,200 queries/sec |
| ArguAna | 8,674 docs | 3.5s | 68ms | 1,470 queries/sec |
| Climate-FEVER | 10K docs | 4.2s | 82ms | 1,220 queries/sec |

**GPU Performance** (NVIDIA T4):
- Encoding: ~50K sentences/sec (batch size 64)
- Query latency: 15-20ms (3x faster than CPU)

## Strengths and Weaknesses

### Strengths ✅

1. **Semantic understanding**: Captures meaning, not just keywords
2. **Paraphrase handling**: Matches semantically similar text
3. **Vocabulary independence**: Works across synonym/paraphrase variations
4. **Cross-lingual**: Multilingual models support multiple languages
5. **Context-aware**: Understands context and relationships
6. **State-of-the-art**: Best performance on semantic search benchmarks

### Weaknesses ❌

1. **Slower than BM25**: Requires neural network inference
2. **GPU recommended**: CPU encoding is 5-10x slower
3. **Less interpretable**: Hard to explain why documents match
4. **Misses exact matches**: May not prioritize exact keyword/phrase matches
5. **Resource intensive**: Requires 500MB-2GB model in memory
6. **Quality depends on model**: Needs appropriate pre-training domain

## When to Use Dense Retrieval

### Best Use Cases

✅ **Semantic search**: Understanding meaning beyond keywords
✅ **Paraphrased queries**: "how to cook pasta" → "pasta preparation"
✅ **Question answering**: Natural language questions
✅ **Multi-lingual search**: Query in English, find French documents
✅ **Conceptual queries**: "renewable energy" → documents about solar, wind, hydro
✅ **Recommendation**: Finding similar documents/products

### Not Ideal For

❌ **Exact keyword matching**: "find API key configuration" (use BM25)
❌ **Entity search**: "John Smith from Microsoft" (lexical better)
❌ **Very short queries**: "ML" (ambiguous, lacks context)
❌ **Domain mismatch**: General model on specialized domain (medical, legal)
❌ **Real-time constraints**: Sub-5ms latency requirements

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Dim | Speed | Quality |
|----------|-------------------|-----|-------|---------|
| **General purpose** | all-MiniLM-L6-v2 | 384 | Fast | Good |
| **High quality** | all-mpnet-base-v2 | 768 | Medium | Excellent |
| **Multi-lingual** | paraphrase-multilingual-MiniLM-L12-v2 | 384 | Fast | Good |
| **Question answering** | multi-qa-mpnet-base-dot-v1 | 768 | Medium | Excellent |
| **Code search** | code-search-net | 768 | Medium | Good |
| **Scientific papers** | allenai-specter | 768 | Medium | Excellent |

### By Performance Needs

**Maximum Speed**:
```python
# Smallest, fastest model
retriever = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
)
```

**Balanced**:
```python
# Good balance of speed and quality
retriever = DenseRetriever(
    model_name="sentence-transformers/all-MiniLM-L12-v2"  # 384-dim
)
```

**Maximum Quality**:
```python
# Best quality, slower
retriever = DenseRetriever(
    model_name="sentence-transformers/all-mpnet-base-v2"  # 768-dim
)
```

## Integration with Hybrid Retrieval

Dense retrieval is most effective when combined with BM25:

```python
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever

# Initialize both retrievers
bm25 = BM25Retriever()
bm25.fit(chunks)

dense = DenseRetriever(
    vector_store=vector_store,
    model_name="all-MiniLM-L6-v2"
)

# Hybrid combines both with Reciprocal Rank Fusion
hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    k_rrf=60
)

# Best of both: keyword + semantic
result = hybrid.retrieve("machine learning tutorial", k=10)
```

**Benchmark Results** (SciFact):
- BM25 alone: NDCG@10 = 0.62
- Dense alone: NDCG@10 = 0.74
- **Hybrid (BM25 + Dense): NDCG@10 = 0.78** ✅

## Advanced Topics

### Fine-Tuning on Domain Data

For specialized domains, fine-tune the embedding model:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data (query, positive doc, negative doc)
train_examples = [
    InputExample(texts=["query 1", "relevant doc", "irrelevant doc"]),
    InputExample(texts=["query 2", "relevant doc", "irrelevant doc"]),
    # ...
]

# Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define loss (triplet loss for ranking)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# Save fine-tuned model
model.save("models/finetuned-dense-retriever")

# Use in retriever
retriever = DenseRetriever(model_name="models/finetuned-dense-retriever")
```

### Hard Negative Mining

Improve model by training with hard negatives (similar but irrelevant docs):

```python
# During fine-tuning, use BM25 to find hard negatives
hard_negatives = bm25.retrieve(query, k=20)  # Get BM25 top-20
hard_negatives = [doc for doc in hard_negatives if doc not in relevant_docs]

# Use as training signal
train_example = InputExample(
    texts=[query, relevant_doc, hard_negatives[0]]
)
```

### Query Expansion

Expand queries for better retrieval:

```python
def expand_query(query: str, expansion_model) -> str:
    # Generate semantically related terms
    expanded = expansion_model.generate(
        f"Expand this query: {query}",
        max_length=50
    )
    return f"{query} {expanded}"

# Use expanded query
expanded_query = expand_query("machine learning", llm)
result = retriever.retrieve(expanded_query, k=10)
```

### Embedding Caching

Cache embeddings to avoid recomputation:

```python
import pickle
from pathlib import Path

# Encode and cache
embeddings_cache = {}
for chunk in chunks:
    key = chunk.id
    if key not in embeddings_cache:
        embeddings_cache[key] = model.encode(chunk.text)

# Save cache
with open("embeddings_cache.pkl", "wb") as f:
    pickle.dump(embeddings_cache, f)

# Load cache
with open("embeddings_cache.pkl", "rb") as f:
    embeddings_cache = pickle.load(f)

# Use cached embeddings
for chunk in chunks:
    chunk.embedding = embeddings_cache[chunk.id]
```

## Troubleshooting

### Issue: Slow Query Performance

**Cause**: CPU encoding or brute-force search

**Solutions**:
```python
# 1. Use GPU
device = "cuda"
retriever.embedding_model.model.to(device)

# 2. Use approximate search (FAISS)
from dartboard.storage.vector_store import FAISSVectorStore
vector_store = FAISSVectorStore(embedding_dim=384, use_gpu=True)
vector_store.build_index(index_type="IVF")  # Approximate, faster

# 3. Reduce embedding dimension
retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")  # 384 vs 768
```

### Issue: Poor Results on Domain-Specific Queries

**Cause**: Model not trained on domain data

**Solutions**:
```python
# 1. Use domain-specific model
retriever = DenseRetriever(
    model_name="allenai-specter"  # For scientific papers
)

# 2. Fine-tune on your domain data
# (See Fine-Tuning section above)

# 3. Combine with BM25 (hybrid)
hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
```

### Issue: Missing Exact Keyword Matches

**Cause**: Dense embeddings prioritize semantic similarity

**Solution**: Use hybrid retrieval
```python
from dartboard.retrieval.hybrid import HybridRetriever
hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    weight_bm25=0.6,  # Increase BM25 weight for exact matches
    weight_dense=0.4
)
```

### Issue: Out of Memory

**Cause**: Large corpus or high embedding dimension

**Solutions**:
```python
# 1. Use smaller model
retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")  # 384-dim

# 2. Process in batches
batch_size = 1000
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    embeddings = retriever.encode_batch([c.text for c in batch])
    vector_store.add(batch)

# 3. Use cloud vector store (Pinecone)
from dartboard.storage.vector_store import PineconeVectorStore
vector_store = PineconeVectorStore(...)  # Offload to cloud
```

## References

### Key Papers

1. **Dense Passage Retrieval (DPR)**: Karpukhin et al. (2020)
   - Introduced dense retrieval for open-domain QA
   - BERT-based bi-encoder architecture

2. **Sentence-BERT**: Reimers & Gurevych (2019)
   - Efficient sentence embeddings with Siamese networks
   - Foundation for modern dense retrieval

3. **ColBERT**: Khattab & Zaharia (2020)
   - Late interaction for better efficiency-quality tradeoff

### Libraries and Models

- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **Model Hub**: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
- **FAISS**: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

### Implementation

- **Source Code**: [dartboard/retrieval/dense.py](../dartboard/retrieval/dense.py)
- **Tests**: [tests/test_dense.py](../tests/test_dense.py)
- **Benchmarks**: [benchmarks/README.md](../benchmarks/README.md)

## Summary

Dense retrieval provides **powerful semantic search** capabilities through learned embeddings. It excels at understanding meaning, handling paraphrased queries, and working across languages, making it essential for modern RAG systems.

**Key Takeaways**:
- ✅ Use dense retrieval for semantic search and question answering
- ✅ all-MiniLM-L6-v2 is a good default model (fast, quality)
- ✅ Combine with BM25 in hybrid mode for best results
- ✅ GPU acceleration provides 5-10x speedup
- ✅ FAISS enables efficient similarity search at scale
- ❌ Dense alone may miss exact keyword/entity matches (use hybrid)
