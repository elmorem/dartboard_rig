# Rerankers and Two-Stage Retrieval

## Overview

**Reranking** is the process of rescoring a set of candidate documents with a more sophisticated (and slower) model to improve relevance ranking. In RAG systems, reranking enables the use of high-quality models (cross-encoders) on a small candidate set retrieved by a fast first-stage retriever (BM25 or dense).

## Why Reranking?

### The Speed-Quality Dilemma

| Retriever Type | Speed | Quality | Scalability |
|----------------|-------|---------|-------------|
| **BM25** | Very Fast | Medium | Excellent (millions of docs) |
| **Dense (Bi-Encoder)** | Fast | Good | Good (FAISS index) |
| **Cross-Encoder** | Slow | Excellent | Poor (O(N) rescoring) |

**Solution**: Two-stage pipeline
1. **Stage 1 (Retrieval)**: Fast method retrieves 100-1000 candidates
2. **Stage 2 (Reranking)**: Slow, high-quality method rescores top candidates

## Two-Stage Retrieval Pipeline

```
Query
  ↓
[Stage 1: Fast Retrieval]
  BM25 or Dense Retrieval
  → Retrieve top 100-1000 candidates (~10-50ms)
  ↓
[Stage 2: Reranking]
  Cross-Encoder or Dartboard
  → Rerank to top 5-10 results (~100-500ms)
  ↓
Final Results (high quality, manageable latency)
```

### Example Implementation

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.embeddings import CrossEncoder

# Stage 1: Dense retrieval (fast)
dense_retriever = DenseRetriever(vector_store=vector_store)
candidates = dense_retriever.retrieve(query, k=100)
# Latency: ~50ms, Quality: Good

# Stage 2: Cross-encoder reranking (slow but accurate)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
texts = [chunk.text for chunk in candidates.chunks]
scores = cross_encoder.score(query, texts)
# Latency: ~200ms, Quality: Excellent

# Sort by reranked scores
for chunk, score in zip(candidates.chunks, scores):
    chunk.score = score

candidates.chunks.sort(key=lambda c: c.score, reverse=True)
final_results = candidates.chunks[:10]  # Top 10
```

## Reranking Methods in This Repository

### 1. Cross-Encoder Reranking (Highest Quality)

```python
from dartboard.embeddings import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Rerank candidates
scores = cross_encoder.score(query, candidate_texts)
reranked_indices = np.argsort(scores)[::-1]
reranked_docs = [candidates[i] for i in reranked_indices]
```

**Pros**:
- Highest quality (attends to both query and doc)
- State-of-the-art on benchmarks
- Fine-grained relevance scoring

**Cons**:
- Slow (100-200ms for 100 candidates)
- GPU recommended
- Cannot pre-compute

**Best For**: High-precision applications, final reranking stage

### 2. Dartboard Reranking (Diversity-Aware)

```python
from dartboard.core import DartboardRetriever, DartboardConfig

# Configure Dartboard for reranking
config = DartboardConfig(
    sigma=1.0,
    top_k=10,
    triage_k=100,  # Already have 100 candidates
    reranker_type="hybrid"  # Cross-encoder + cosine
)

retriever = DartboardRetriever(config, embedding_model, cross_encoder)

# Rerank with diversity
result = retriever.retrieve(query, candidates)
# Returns diverse, non-redundant top-10
```

**Pros**:
- Balances relevance and diversity
- Automatic diversity (no λ parameter)
- Works with cross-encoder or cosine

**Cons**:
- Slower than pure cross-encoder (greedy selection)
- More complex

**Best For**: Multi-aspect queries, exploratory search

### 3. Hybrid Retrieval (BM25 + Dense Fusion)

```python
from dartboard.retrieval.hybrid import HybridRetriever

# Combines BM25 and Dense with RRF
hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    k_rrf=60
)

# Retrieves and reranks in one step
result = hybrid.retrieve(query, k=10)
```

**Pros**:
- Combines lexical + semantic
- RRF fusion is parameter-free
- Outperforms either method alone

**Cons**:
- Runs both retrievers (2x cost)
- Not as high-quality as cross-encoder

**Best For**: General-purpose retrieval, balancing speed and quality

## Reranking Strategies

### Strategy 1: Cross-Encoder Only

```python
# Simple and effective
dense_results = dense_retriever.retrieve(query, k=100)
reranked = cross_encoder_rerank(query, dense_results)
top_results = reranked[:10]
```

**Use When**: Quality is paramount, latency <500ms acceptable

### Strategy 2: Dartboard with Cross-Encoder

```python
# Diversity + quality
config = DartboardConfig(reranker_type="crossencoder")
dartboard = DartboardRetriever(config, embedding_model, cross_encoder)
result = dartboard.retrieve(query, candidates)
```

**Use When**: Need diverse results without redundancy

### Strategy 3: Cascading Rerankers

```python
# Stage 1: Dense retrieval (1000 candidates)
candidates = dense_retriever.retrieve(query, k=1000)

# Stage 2: Fast reranker (1000 → 100)
fast_reranked = cosine_rerank(query, candidates, k=100)

# Stage 3: Slow reranker (100 → 10)
final_results = cross_encoder_rerank(query, fast_reranked, k=10)
```

**Use When**: Very large corpus, need maximum quality

### Strategy 4: Hybrid + Cross-Encoder

```python
# Stage 1: Hybrid retrieval (BM25 + Dense)
hybrid_results = hybrid_retriever.retrieve(query, k=100)

# Stage 2: Cross-encoder reranking
final_results = cross_encoder_rerank(query, hybrid_results, k=10)
```

**Use When**: Want both lexical/semantic + high quality

## Performance Comparison (SciFact Benchmark)

| Method | NDCG@10 | Latency (p95) | Notes |
|--------|---------|---------------|-------|
| BM25 alone | 0.62 | 12ms | Fast, keyword-based |
| Dense alone | 0.74 | 45ms | Good semantic matching |
| Hybrid (BM25 + Dense) | 0.78 | 68ms | Best of both |
| **Dense + Cross-Encoder** | **0.82** | 250ms | Highest quality |
| Dense + Dartboard | 0.71 | 95ms | More diverse |

**Conclusion**: Cross-encoder reranking provides +8% NDCG improvement over hybrid at 3.5x latency cost.

## When to Use Which Reranker

### Use Cross-Encoder When:
- ✅ Quality is critical (medical, legal, financial)
- ✅ Candidate set is small (10-100 docs)
- ✅ Latency budget allows 100-500ms
- ✅ Have GPU for inference

### Use Dartboard When:
- ✅ Need diversity (multi-aspect queries)
- ✅ Want to avoid redundant results
- ✅ Exploring a topic comprehensively

### Use Hybrid When:
- ✅ Balanced speed and quality
- ✅ General-purpose retrieval
- ✅ No reranking budget

## Implementation Patterns

### Pattern: Progressive Refinement

```python
# Start broad, refine progressively
k1, k2, k3 = 1000, 100, 10

# Stage 1: Fast retrieval
candidates = bm25_retriever.retrieve(query, k=k1)

# Stage 2: Dense reranking
dense_scores = dense_similarity(query_emb, candidates)
candidates = top_k_by_score(candidates, dense_scores, k=k2)

# Stage 3: Cross-encoder reranking
ce_scores = cross_encoder.score(query, candidates)
final = top_k_by_score(candidates, ce_scores, k=k3)
```

### Pattern: Ensemble Reranking

```python
# Combine multiple rerankers
bm25_scores = normalize(bm25_retriever.retrieve(query, k=100).scores)
dense_scores = normalize(dense_retriever.retrieve(query, k=100).scores)
ce_scores = normalize(cross_encoder.score(query, candidates))

# Weighted ensemble
ensemble_scores = (
    0.2 * bm25_scores +
    0.3 * dense_scores +
    0.5 * ce_scores
)

final_results = top_k_by_score(candidates, ensemble_scores, k=10)
```

## Optimization Tips

### 1. Batch Reranking

```python
# Rerank multiple queries efficiently
queries = ["query1", "query2", "query3"]
all_candidates = [retrieve(q, k=100) for q in queries]

# Batch all pairs
all_pairs = []
for query, candidates in zip(queries, all_candidates):
    all_pairs.extend([(query, c.text) for c in candidates])

# Single batch inference
all_scores = cross_encoder.model.predict(all_pairs, batch_size=64)

# Split back into per-query results
idx = 0
for candidates in all_candidates:
    scores = all_scores[idx:idx+len(candidates)]
    # ... rank candidates by scores ...
    idx += len(candidates)
```

### 2. Caching

```python
# Cache reranking scores for common queries
reranking_cache = {}

def cached_rerank(query, candidates):
    cache_key = hash((query, tuple(c.id for c in candidates)))

    if cache_key in reranking_cache:
        return reranking_cache[cache_key]

    scores = cross_encoder.score(query, [c.text for c in candidates])
    reranking_cache[cache_key] = scores
    return scores
```

### 3. Early Stopping

```python
# Stop reranking if top-k are clearly dominant
def rerank_with_early_stop(query, candidates, k=10, threshold=0.95):
    scores = []

    for i, candidate in enumerate(candidates):
        score = cross_encoder.score(query, [candidate.text])[0]
        scores.append((candidate, score))

        # Early stop if top-k have very high scores
        if i >= k:
            top_k_scores = sorted([s for _, s in scores], reverse=True)[:k]
            if min(top_k_scores) > threshold:
                break  # Top-k are confident

    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
```

## Troubleshooting

### Issue: Reranking Too Slow

**Solutions**:
```python
# 1. Reduce candidate set
candidates = dense_retriever.retrieve(query, k=50)  # Instead of 100

# 2. Use GPU
cross_encoder = CrossEncoder(model_name, device="cuda")

# 3. Use smaller/faster model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# 4. Batch inference
scores = cross_encoder.model.predict(pairs, batch_size=64)
```

### Issue: Reranking Hurts Performance

**Cause**: Reranker has different training domain than retrieval task

**Solutions**:
```python
# 1. Use ensemble (don't rely 100% on reranker)
final_score = 0.7 * retrieval_score + 0.3 * reranking_score

# 2. Fine-tune reranker on your data

# 3. Try different reranking model
cross_encoder = CrossEncoder("cross-encoder/qnli-electra-base")
```

## References

### Papers

1. **Contextualized Reranking**: Nogueira & Cho (2019) - "Passage Re-ranking with BERT"
2. **Dense Passage Retrieval**: Karpukhin et al. (2020)
3. **RRF Fusion**: Cormack et al. (2009) - "Reciprocal Rank Fusion"

### Implementation

- **Cross-Encoder**: [dartboard/embeddings.py](../dartboard/embeddings.py)
- **Hybrid Retrieval**: [dartboard/retrieval/hybrid.py](../dartboard/retrieval/hybrid.py)
- **Dartboard**: [dartboard/core.py](../dartboard/core.py)

## Summary

Reranking is essential for high-quality RAG. The **two-stage pipeline** (fast retrieval + cross-encoder reranking) provides the best balance of speed and quality for most applications.

**Key Takeaways**:
- ✅ Use **two-stage retrieval**: Fast retriever (100 candidates) → Cross-encoder reranking (top 10)
- ✅ **Cross-encoder reranking** provides +8-10% NDCG improvement
- ✅ **GPU acceleration** is critical for cross-encoder performance
- ✅ Combine with **Dartboard** for diversity-aware reranking
- ✅ **Hybrid retrieval** (BM25 + Dense) is good baseline before reranking
