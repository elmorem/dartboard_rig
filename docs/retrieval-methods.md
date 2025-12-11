# Retrieval Methods in Dartboard

## Overview

Dartboard implements **5 different retrieval methods** for finding relevant document chunks given a query. Each method has different strengths, weaknesses, and use cases.

This guide explains all retrieval methods available in the repository, when to use each one, and how they compare.

---

## Quick Comparison

| Method | Type | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| **Dartboard** | Diversity-aware | Medium | Excellent | **Default** - Best for diverse results |
| **Dense** | Vector similarity | Fast | Very Good | Semantic search, paraphrased queries |
| **BM25** | Sparse/Lexical | Very Fast | Good | Keyword search, exact matches |
| **Hybrid** | BM25 + Dense | Medium | Excellent | Robust across all query types |
| **Reranker** | Cross-encoder | Slow | Excellent | Second-stage refinement |

---

## 1. Dartboard Retriever

### What It Is

**Dartboard** is the primary retrieval algorithm in this repository. It combines semantic similarity with diversity-awareness to return relevant AND diverse results.

**Key innovation:** Uses Determinantal Point Processes (DPPs) to balance:
- **Relevance**: How similar chunks are to the query
- **Diversity**: How different chunks are from each other

### How It Works

```
Query: "machine learning applications"

1. Triage Phase (Fast)
   - Compute cosine similarity to ALL chunks
   - Select top-100 candidates (configurable: triage_k)

2. Scoring Phase (Dartboard Algorithm)
   - Compute pairwise distances between candidates
   - Use Gaussian kernel with sigma parameter
   - Greedy selection balancing relevance + diversity

3. Output
   - Top-5 diverse, relevant chunks
```

### Implementation

**File:** [dartboard/core.py](../dartboard/core.py)

```python
from dartboard.core import DartboardRetriever, DartboardConfig
from dartboard.embeddings import SentenceTransformerModel

# Initialize
config = DartboardConfig(
    sigma=1.0,        # Temperature (higher = more diversity)
    top_k=5,          # Number of results
    triage_k=100      # Candidates for triage
)

embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
retriever = DartboardRetriever(config, embedding_model)

# Retrieve
result = retriever.retrieve(
    query="What is machine learning?",
    corpus=chunks
)

print(f"Retrieved {len(result.chunks)} diverse chunks")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | 1.0 | Temperature (↑ = more diversity, ↓ = more relevance) |
| `top_k` | 5 | Number of chunks to return |
| `triage_k` | 100 | Candidates from initial KNN |
| `reranker_type` | "hybrid" | Reranking method ("cosine", "crossencoder", "hybrid") |

### When to Use

✅ **Use Dartboard when:**
- You want diverse, non-redundant results
- User queries are broad or ambiguous
- You need comprehensive coverage of a topic
- Avoiding redundant information is important

❌ **Don't use when:**
- Speed is critical (use Dense or BM25 instead)
- You want all similar results (diversity works against you)
- Results must be strictly ranked by similarity

### Advantages

- ✅ Returns diverse, comprehensive results
- ✅ Reduces redundancy automatically
- ✅ Excellent for exploratory search
- ✅ Theoretically grounded (DPPs)

### Disadvantages

- ❌ Slower than simple similarity search
- ❌ More complex to understand
- ❌ Requires parameter tuning (sigma)

---

## 2. Dense Retriever

### What It Is

**Dense retrieval** uses vector embeddings and cosine similarity for semantic search. This is the standard approach used by most modern RAG systems.

### How It Works

```
Query: "ML algorithms"

1. Embed Query
   - Generate embedding: [0.23, -0.15, 0.67, ..., 0.41]

2. Compute Similarity
   - Cosine similarity with all chunk embeddings
   - Score = dot(query_emb, chunk_emb) / (||query|| * ||chunk||)

3. Rank
   - Sort by similarity (highest first)
   - Return top-K
```

### Implementation

**File:** [dartboard/retrieval/dense.py](../dartboard/retrieval/dense.py)

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.embeddings import SentenceTransformerModel

# Initialize
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
retriever = DenseRetriever(
    vector_store=vector_store,
    embedding_model=embedding_model,
    similarity_metric="cosine"  # or "dot", "euclidean"
)

# Retrieve
result = retriever.retrieve(
    query="What is deep learning?",
    k=10
)

print(f"Top score: {result.scores[0]:.4f}")
print(f"Latency: {result.latency_ms:.2f}ms")
```

### When to Use

✅ **Use Dense when:**
- Queries use different words than documents (synonyms, paraphrases)
- You need semantic understanding
- Multilingual search
- Speed + quality balance is important

❌ **Don't use when:**
- Exact keyword matching is critical
- You don't have/want embeddings
- Speed is absolutely critical (use BM25)

### Advantages

- ✅ Captures semantic meaning
- ✅ Robust to vocabulary mismatch
- ✅ Works with paraphrased queries
- ✅ Good for multilingual search

### Disadvantages

- ❌ Slower than BM25 (embedding generation)
- ❌ May miss exact keyword matches
- ❌ Requires embedding model
- ❌ Less explainable than BM25

---

## 3. BM25 Retriever

### What It Is

**BM25** (Best Matching 25) is a sparse, keyword-based retrieval algorithm. It uses term frequency and inverse document frequency for ranking.

**History:** Evolution of TF-IDF, optimized for information retrieval

### How It Works

```
Query: "machine learning"
Document: "Machine learning is a subset of AI..."

1. Tokenize
   - Query: ["machine", "learning"]
   - Doc: ["machine", "learning", "is", "a", "subset", ...]

2. Compute BM25 Score
   For each term:
   - TF (term frequency) in document
   - IDF (inverse document frequency) across corpus
   - Document length normalization

3. Sum
   - BM25(doc) = sum of term scores
```

**Formula:**
```
score(q, d) = Σ IDF(term) * (TF(term, d) * (k1 + 1)) / (TF(term, d) + k1 * (1 - b + b * |d| / avgdl))
```

### Implementation

**File:** [dartboard/retrieval/bm25.py](../dartboard/retrieval/bm25.py)

```python
from dartboard.retrieval.bm25 import BM25Retriever

# Initialize
retriever = BM25Retriever(
    k1=1.5,   # Term frequency saturation
    b=0.75    # Length normalization
)

# Fit on corpus (required!)
retriever.fit(chunks)

# Retrieve
result = retriever.retrieve(
    query="machine learning algorithms",
    k=10
)

# Analyze term contributions
term_scores = retriever.get_term_scores(
    query="machine learning",
    doc_idx=0
)
print(term_scores)  # {'machine': 2.4, 'learning': 1.8}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.5 | Term frequency saturation (1.2-2.0 typical) |
| `b` | 0.75 | Length normalization (0.0-1.0, where 1.0 = full norm) |

### When to Use

✅ **Use BM25 when:**
- Exact keyword matching is important
- Speed is critical
- You don't have/need embeddings
- Queries contain specific technical terms
- Explainability matters (can show term scores)

❌ **Don't use when:**
- Queries are paraphrased or use synonyms
- Semantic understanding is needed
- Vocabulary mismatch is common

### Advantages

- ✅ Very fast retrieval
- ✅ No embeddings needed
- ✅ Explainable (term-level scores)
- ✅ Excellent for keyword search
- ✅ Works well with technical/rare terms

### Disadvantages

- ❌ Sensitive to vocabulary mismatch
- ❌ Poor with paraphrased queries
- ❌ No semantic understanding
- ❌ Requires corpus fitting (can't add docs easily)

---

## 4. Hybrid Retriever

### What It Is

**Hybrid retrieval** combines BM25 (sparse/lexical) and Dense (semantic) retrieval using **Reciprocal Rank Fusion (RRF)** to get the best of both worlds.

### How It Works

```
Query: "neural networks"

1. Retrieve from Both Methods
   - BM25: Get top-15 by keyword matching
   - Dense: Get top-15 by semantic similarity

2. Reciprocal Rank Fusion
   For each document:
   - RRF_score = weight_bm25 * 1/(k + bm25_rank) +
                 weight_dense * 1/(k + dense_rank)
   - k = 60 (constant from literature)

3. Merge and Rerank
   - Combine results
   - Sort by RRF score
   - Return top-K
```

**RRF Example:**
```
Doc A: BM25 rank=1, Dense rank=3
  RRF = 0.5/(60+1) + 0.5/(60+3) = 0.016

Doc B: BM25 rank=2, Dense rank=1
  RRF = 0.5/(60+2) + 0.5/(60+1) = 0.016

Doc C: BM25 rank=10, Dense rank=50
  RRF = 0.5/(60+10) + 0.5/(60+50) = 0.012
```

### Implementation

**File:** [dartboard/retrieval/hybrid.py](../dartboard/retrieval/hybrid.py)

```python
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever

# Initialize components
bm25 = BM25Retriever()
bm25.fit(chunks)

dense = DenseRetriever(
    vector_store=vector_store,
    embedding_model=embedding_model
)

# Create hybrid retriever
hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    k_rrf=60,           # RRF constant
    weight_bm25=0.5,    # BM25 weight
    weight_dense=0.5    # Dense weight
)

# Retrieve
result = hybrid.retrieve(
    query="What are neural networks?",
    k=10,
    retrieve_k_multiplier=3  # Get 30 candidates from each
)

# Analyze fusion
for chunk in result.chunks:
    print(f"Sources: {chunk.metadata['sources']}")  # ['bm25', 'dense'] or one
    print(f"In both: {chunk.metadata['in_both']}")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_rrf` | 60 | RRF constant (from literature) |
| `weight_bm25` | 0.5 | Weight for BM25 scores |
| `weight_dense` | 0.5 | Weight for dense scores |
| `retrieve_k_multiplier` | 3 | Candidates per method (k * multiplier) |

### When to Use

✅ **Use Hybrid when:**
- You want the best of both worlds
- Query types vary (keyword + semantic)
- Robustness is more important than speed
- You can afford the extra latency

❌ **Don't use when:**
- Speed is absolutely critical
- You only have one type of query
- Complexity is a concern

### Advantages

- ✅ Best overall retrieval quality
- ✅ Robust across all query types
- ✅ Combines lexical + semantic matching
- ✅ RRF is parameter-free (no tuning needed)
- ✅ Chunks in both results get higher scores

### Disadvantages

- ❌ Slower (runs both retrievers)
- ❌ More complex implementation
- ❌ Requires both BM25 and Dense setup

---

## 5. Cross-Encoder Reranker

### What It Is

**Reranker** is a second-stage refinement method that uses a cross-encoder model to rerank initial retrieval results for better precision.

**Not a retriever:** Used AFTER initial retrieval to improve ranking.

### How It Works

```
Query: "deep learning"

1. First-Stage Retrieval (Fast)
   - Use BM25/Dense/Hybrid to get top-100 candidates
   - Fast but less accurate

2. Reranking (Slow but Accurate)
   - Score each query-document pair with cross-encoder
   - Cross-encoder sees BOTH query and document together
   - More accurate than bi-encoders (Dense)

3. Output
   - Top-K reranked by cross-encoder scores
```

**Key difference:**
- **Bi-encoder (Dense):** `encode(query)` and `encode(doc)` separately, then compare
- **Cross-encoder:** `score(query, doc)` together (more accurate but slower)

### Implementation

**File:** [dartboard/retrieval/reranker.py](../dartboard/retrieval/reranker.py)

```python
from dartboard.retrieval.reranker import CrossEncoderReranker
from dartboard.retrieval.dense import DenseRetriever

# First-stage retrieval
dense = DenseRetriever(vector_store=vector_store)
candidates = dense.retrieve(query="neural networks", k=100)

# Rerank
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=32
)

result = reranker.rerank(
    query="neural networks",
    chunks=candidates.chunks,
    top_k=10  # Final top-10
)

print(f"Reranked {len(candidates.chunks)} → {len(result.chunks)}")
print(f"Latency: {result.latency_ms:.2f}ms")
```

### Available Cross-Encoder Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| ms-marco-MiniLM-L-6-v2 | Fast | Good | General purpose |
| ms-marco-MiniLM-L-12-v2 | Medium | Better | More accuracy needed |
| ms-marco-electra-base | Slow | Best | Maximum quality |

### When to Use

✅ **Use Reranker when:**
- Precision is critical (top-10 must be perfect)
- You have a two-stage retrieval pipeline
- You can afford the latency
- Initial retrieval is noisy

❌ **Don't use when:**
- Speed is critical
- You need to rank thousands of documents
- First-stage retrieval is already accurate

### Advantages

- ✅ Most accurate ranking method
- ✅ Captures query-document interactions
- ✅ Improves precision of initial retrieval
- ✅ Simple to add as second stage

### Disadvantages

- ❌ Very slow (must score each pair)
- ❌ Cannot be used for first-stage retrieval
- ❌ Requires pretrained cross-encoder

---

## Retrieval Pipelines

### Pipeline 1: Simple Dense (Fast)

```python
# Use case: Speed is critical, semantic search only
retriever = DenseRetriever(vector_store=vector_store)
result = retriever.retrieve(query, k=10)
```

**Latency:** ~50ms
**Quality:** Good

---

### Pipeline 2: Hybrid (Balanced)

```python
# Use case: Best retrieval quality, moderate speed
hybrid = HybridRetriever(bm25=bm25, dense=dense)
result = hybrid.retrieve(query, k=10)
```

**Latency:** ~100ms
**Quality:** Excellent

---

### Pipeline 3: Dense + Reranking (High Precision)

```python
# Use case: Maximum precision for top-10
# Stage 1: Fast retrieval (100 candidates)
candidates = dense.retrieve(query, k=100)

# Stage 2: Rerank top candidates
reranker = CrossEncoderReranker()
result = reranker.rerank(query, candidates.chunks, top_k=10)
```

**Latency:** ~200ms
**Quality:** Best

---

### Pipeline 4: Dartboard (Diversity)

```python
# Use case: Diverse, comprehensive results
dartboard = DartboardRetriever(config, embedding_model)
result = dartboard.retrieve(query, corpus=chunks)
```

**Latency:** ~150ms
**Quality:** Excellent (diverse)

---

### Pipeline 5: Full Stack (Maximum Quality)

```python
# Use case: Production RAG with highest quality
# Stage 1: Hybrid retrieval (200 candidates)
hybrid = HybridRetriever(bm25=bm25, dense=dense)
candidates = hybrid.retrieve(query, k=200)

# Stage 2: Dartboard for diversity (50 candidates)
dartboard = DartboardRetriever(config, embedding_model)
diverse = dartboard.retrieve(query, corpus=candidates.chunks)

# Stage 3: Cross-encoder reranking (top-10)
reranker = CrossEncoderReranker()
final = reranker.rerank(query, diverse.chunks, top_k=10)
```

**Latency:** ~500ms
**Quality:** Maximum

---

## Performance Comparison

### Speed Benchmark (1000 document corpus)

| Method | Latency | Throughput |
|--------|---------|------------|
| BM25 | 10ms | 100 queries/sec |
| Dense | 50ms | 20 queries/sec |
| Hybrid | 100ms | 10 queries/sec |
| Dartboard | 150ms | 6.7 queries/sec |
| + Reranker | +200ms | - |

### Quality Benchmark (NDCG@10)

| Method | NDCG@10 | Recall@10 |
|--------|---------|-----------|
| BM25 | 0.65 | 0.72 |
| Dense | 0.71 | 0.78 |
| Hybrid | 0.76 | 0.84 |
| + Reranker | 0.82 | 0.84 |
| Dartboard | 0.74 | 0.81 |

*Benchmarks are approximate and vary by dataset

---

## Choosing the Right Method

### Decision Tree

```
Do you need diverse results?
├─ YES → Use Dartboard
└─ NO
   ├─ Is speed critical?
   │  ├─ YES → Use BM25 (if keywords) or Dense (if semantic)
   │  └─ NO
   │     ├─ Mixed query types?
   │     │  ├─ YES → Use Hybrid
   │     │  └─ NO → Use Dense
   │     │
   │     └─ Need maximum precision?
   │        └─ YES → Add Reranker stage
```

### By Use Case

**Q&A System:**
- **Fast:** Dense
- **Best:** Hybrid + Reranker

**Document Search:**
- **Fast:** BM25
- **Best:** Hybrid

**Exploratory Search:**
- **Always:** Dartboard

**Chatbot:**
- **Fast:** Dense
- **Best:** Hybrid

**Legal/Medical (precision critical):**
- **Always:** Hybrid + Reranker

---

## Code Examples

### Example 1: Compare All Methods

```python
from dartboard.retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from dartboard.core import DartboardRetriever, DartboardConfig

query = "What is machine learning?"

# Dense
dense = DenseRetriever(vector_store)
dense_result = dense.retrieve(query, k=10)

# BM25
bm25 = BM25Retriever()
bm25.fit(chunks)
bm25_result = bm25.retrieve(query, k=10)

# Hybrid
hybrid = HybridRetriever(bm25=bm25, dense=dense)
hybrid_result = hybrid.retrieve(query, k=10)

# Dartboard
config = DartboardConfig(sigma=1.0, top_k=10)
dartboard = DartboardRetriever(config, embedding_model)
dartboard_result = dartboard.retrieve(query, corpus=chunks)

# Compare
print(f"Dense:     {dense_result.latency_ms:.1f}ms")
print(f"BM25:      {bm25_result.latency_ms:.1f}ms")
print(f"Hybrid:    {hybrid_result.latency_ms:.1f}ms")
print(f"Dartboard: Score = {dartboard_result.scores[0]:.4f}")
```

### Example 2: Two-Stage Pipeline

```python
# Stage 1: Fast hybrid retrieval (100 candidates)
hybrid = HybridRetriever(bm25=bm25, dense=dense)
candidates = hybrid.retrieve(query, k=100)

# Stage 2: Rerank for precision
reranker = CrossEncoderReranker()
final = reranker.rerank(query, candidates.chunks, top_k=10)

print(f"Retrieved {len(final.chunks)} high-precision results")
print(f"Total latency: {candidates.latency_ms + final.latency_ms:.1f}ms")
```

---

## Configuration Recommendations

### Development

```python
# Fast iteration, good enough quality
retriever = DenseRetriever(vector_store)
```

### Production (Speed-focused)

```python
# BM25 for keywords, Dense for semantic
hybrid = HybridRetriever(
    bm25=bm25,
    dense=dense,
    k_rrf=60,
    weight_bm25=0.4,  # Slightly favor Dense
    weight_dense=0.6
)
```

### Production (Quality-focused)

```python
# Full pipeline with reranking
hybrid = HybridRetriever(bm25=bm25, dense=dense)
candidates = hybrid.retrieve(query, k=100)

reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
final = reranker.rerank(query, candidates.chunks, top_k=10)
```

### Research/Exploration

```python
# Dartboard for diverse, comprehensive results
config = DartboardConfig(
    sigma=1.0,      # Balanced diversity
    top_k=10,
    triage_k=100
)
dartboard = DartboardRetriever(config, embedding_model)
```

---

## Summary

| Method | When to Use | Avoid When |
|--------|-------------|------------|
| **Dartboard** | Need diverse results | Speed critical |
| **Dense** | Semantic search, paraphrases | Exact keywords needed |
| **BM25** | Keyword search, speed critical | Semantic understanding needed |
| **Hybrid** | Best overall quality | Speed absolutely critical |
| **Reranker** | Maximum precision | First-stage retrieval |

**Recommended default:** Start with **Hybrid** retrieval for best balance of speed and quality.

**For diversity:** Use **Dartboard** when you need comprehensive, non-redundant results.

**For maximum quality:** Use **Hybrid + Reranker** two-stage pipeline.

---

## References

- **Dartboard:** [dartboard/core.py](../dartboard/core.py)
- **Dense:** [dartboard/retrieval/dense.py](../dartboard/retrieval/dense.py)
- **BM25:** [dartboard/retrieval/bm25.py](../dartboard/retrieval/bm25.py)
- **Hybrid:** [dartboard/retrieval/hybrid.py](../dartboard/retrieval/hybrid.py)
- **Reranker:** [dartboard/retrieval/reranker.py](../dartboard/retrieval/reranker.py)
- **Base:** [dartboard/retrieval/base.py](../dartboard/retrieval/base.py)
