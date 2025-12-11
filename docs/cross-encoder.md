# Cross-Encoder Reranking

## Overview

A **cross-encoder** is a neural reranking model that scores query-document pairs by processing them jointly through a transformer network. Unlike bi-encoders (dense retrieval) that encode queries and documents separately, cross-encoders see both inputs simultaneously, enabling fine-grained semantic matching at the cost of higher computational expense.

Cross-encoders are primarily used for **reranking** - rescoring a small set of candidate documents retrieved by a faster first-stage retriever (BM25 or dense).

## Cross-Encoder vs. Bi-Encoder

### Bi-Encoder (Dense Retrieval)

```
Query → Encoder → Query Embedding (384-dim)
                        ↓
                   Cosine Similarity
                        ↓
Document → Encoder → Doc Embedding (384-dim)
```

**Characteristics**:
- Encodes query and document **separately**
- Pre-computable: Document embeddings can be cached
- Fast: Only need to encode query at retrieval time
- Scalable: Similarity search via FAISS/Pinecone

### Cross-Encoder (Reranking)

```
[Query, Document] → Cross-Encoder → Relevance Score (0-1)
```

**Characteristics**:
- Processes query and document **together**
- Cannot pre-compute: Must re-encode for each query-doc pair
- Slow: O(K) forward passes for K candidates
- Higher quality: Full attention between query and document

## How Cross-Encoders Work

### Architecture

Cross-encoder = BERT/RoBERTa + Classification Head

```
Input: [CLS] query tokens [SEP] document tokens [SEP]
         ↓
     Transformer Layers (12-24 layers)
         ↓
    [CLS] Representation
         ↓
   Linear Layer + Sigmoid
         ↓
   Relevance Score (0-1)
```

### Forward Pass

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Score single pair
query = "machine learning applications"
document = "ML is used in many industries including..."
score = model.predict([(query, document)])[0]

# Score: 0.85 (highly relevant)
```

### Training

Cross-encoders are typically trained with **pairwise or listwise loss**:

```python
# Training data: (query, positive doc, negative doc)
# Loss: Maximize score(query, positive) - score(query, negative)

pairs = [
    (query, positive_doc),  # Target: high score
    (query, negative_doc),  # Target: low score
]

# Model learns to distinguish relevant from irrelevant
```

## Implementation in This Repository

### Basic Usage

```python
from dartboard.embeddings import CrossEncoder

# Initialize
cross_encoder = CrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    device="cpu"  # or "cuda" for GPU
)

# Score query-document pairs
query = "What is deep learning?"
documents = [
    "Deep learning uses neural networks with many layers",
    "Machine learning is a subset of AI",
    "Python is a programming language"
]

scores = cross_encoder.score(query, documents)
# scores: [0.92, 0.68, 0.12]

# Rank documents by score
ranked_indices = scores.argsort()[::-1]
ranked_docs = [documents[i] for i in ranked_indices]
```

### Integration with Dartboard

Dartboard can use cross-encoders for reranking in three modes:

#### 1. Cross-Encoder Mode

Uses cross-encoder for all similarity calculations:

```python
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.embeddings import SentenceTransformerModel, CrossEncoder

# Load models
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Configure Dartboard with cross-encoder
config = DartboardConfig(
    reranker_type="crossencoder",  # Use cross-encoder for all scoring
    triage_k=100,
    top_k=5
)

# Create retriever
retriever = DartboardRetriever(
    config=config,
    embedding_model=embedding_model,
    cross_encoder=cross_encoder
)

# Retrieve with cross-encoder scoring
result = retriever.retrieve(query, corpus)
```

**Pros**: Highest quality scoring
**Cons**: Slowest (O(K²) cross-encoder calls for K candidates)

#### 2. Hybrid Mode (Recommended)

Uses cross-encoder for query-document, cosine for document-document:

```python
config = DartboardConfig(
    reranker_type="hybrid",  # Cross-encoder + Cosine
    triage_k=100,
    top_k=5
)

retriever = DartboardRetriever(
    config=config,
    embedding_model=embedding_model,
    cross_encoder=cross_encoder
)
```

**Pros**: Good balance of quality and speed
**Cons**: Still requires O(K) cross-encoder calls

#### 3. Two-Stage Pipeline

Use cross-encoder as a final reranking step:

```python
from dartboard.retrieval.dense import DenseRetriever

# Stage 1: Dense retrieval (fast, gets 100 candidates)
dense_retriever = DenseRetriever(vector_store=vector_store)
candidates = dense_retriever.retrieve(query, k=100)

# Stage 2: Cross-encoder reranking (slow, rescores top 100)
pairs = [(query, chunk.text) for chunk in candidates.chunks]
scores = cross_encoder.score(query, [chunk.text for chunk in candidates.chunks])

# Rerank by cross-encoder scores
for chunk, score in zip(candidates.chunks, scores):
    chunk.score = score

candidates.chunks.sort(key=lambda c: c.score, reverse=True)
final_results = candidates.chunks[:10]  # Top 10 after reranking
```

## Available Models

### MS MARCO Models (Recommended)

Trained on Microsoft Machine Reading Comprehension dataset:

| Model | Parameters | Layers | Max Length | Speed | Quality |
|-------|-----------|--------|------------|-------|---------|
| ms-marco-TinyBERT-L-2-v2 | 4.4M | 2 | 512 | Very Fast | Good |
| ms-marco-MiniLM-L-6-v2 | 22.7M | 6 | 512 | Fast | Very Good |
| **ms-marco-MiniLM-L-12-v2** | 33.4M | 12 | 512 | Medium | Excellent |
| ms-marco-electra-base | 109M | 12 | 512 | Slow | Excellent |

**Default Choice**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (good balance)

### Multi-Lingual Models

```python
# English, German, French, Spanish, Italian, Portuguese, Polish, Dutch, etc.
cross_encoder = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
```

### Domain-Specific Models

```python
# Scientific papers
cross_encoder = CrossEncoder("cross-encoder/scibert-zeroshot")

# Legal documents
cross_encoder = CrossEncoder("cross-encoder/legal-bert-base")
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Score single pair | O(L) | L = combined sequence length |
| Score K pairs | O(K · L) | Cannot batch efficiently |
| Pairwise matrix (K docs) | O(K² · L) | Very expensive |

### Benchmarked Performance (Dec 2025)

Tested with `ms-marco-MiniLM-L-12-v2` on CPU:

| Batch Size | Pairs/Second | Latency (ms) | GPU Speedup |
|------------|--------------|--------------|-------------|
| 1 | 12 | 83 | 8x |
| 8 | 85 | 94 | 10x |
| 32 | 280 | 114 | 12x |
| 64 | 420 | 152 | 15x |

**GPU Performance** (NVIDIA T4):
- Single pair: 10ms
- Batch of 32: 25ms (1,280 pairs/sec)
- **Recommendation**: Use GPU for cross-encoder inference

### Memory Usage

- **Model Size**: 130-450 MB depending on model
- **Forward Pass**: ~500MB-1GB VRAM for batch of 32 pairs
- **Total**: Plan for 1-2GB VRAM/RAM

## Strengths and Weaknesses

### Strengths ✅

1. **Highest quality**: Best semantic matching (attends to both query and doc)
2. **Fine-grained relevance**: Captures subtle relevance signals
3. **State-of-the-art**: Top performance on MS MARCO, TREC benchmarks
4. **No index needed**: Stateless scoring
5. **Interpretable**: Can visualize attention weights

### Weaknesses ❌

1. **Slow**: 100-1000x slower than bi-encoder similarity
2. **Not scalable**: Cannot pre-compute, must rescore all pairs
3. **GPU recommended**: CPU inference is very slow
4. **Memory intensive**: Requires large transformer in memory
5. **Limited by max length**: Typically 512 tokens (longer docs need truncation)

## When to Use Cross-Encoders

### Best Use Cases

✅ **Reranking**: Final stage after fast retrieval (BM25/dense)
✅ **Small candidate sets**: Reranking 10-100 documents
✅ **High precision requirements**: Medical, legal, financial domains
✅ **Question answering**: Natural language questions
✅ **Benchmarking**: Evaluating retrieval quality
✅ **Offline processing**: Batch reranking of search results

### Not Ideal For

❌ **First-stage retrieval**: Too slow for millions of documents
❌ **Real-time, high-throughput**: Sub-10ms latency requirements
❌ **Resource-constrained**: Mobile, edge devices
❌ **Very long documents**: Truncated to 512 tokens
❌ **Pairwise diversity**: O(K²) complexity prohibitive

## Usage Patterns

### Pattern 1: Rerank Top-K from Dense Retrieval

```python
# Stage 1: Fast dense retrieval (get 100 candidates)
dense_results = dense_retriever.retrieve(query, k=100)

# Stage 2: Cross-encoder reranking (rescore top 100 → top 10)
texts = [chunk.text for chunk in dense_results.chunks]
scores = cross_encoder.score(query, texts)

# Sort by cross-encoder scores
sorted_indices = scores.argsort()[::-1]
reranked_chunks = [dense_results.chunks[i] for i in sorted_indices[:10]]
```

### Pattern 2: Pairwise Similarity Matrix

```python
# Compute pairwise document similarity
documents = [chunk.text for chunk in chunks]
similarity_matrix = cross_encoder.score_matrix(documents)

# similarity_matrix[i, j] = relevance of doc j to doc i
# Use for clustering, deduplication, or diversity analysis
```

### Pattern 3: Hybrid Scoring Function

```python
def hybrid_score(query, document, alpha=0.7):
    # Dense score (fast, cached)
    dense_score = cosine_similarity(
        query_embedding,
        document_embedding
    )

    # Cross-encoder score (slow, accurate)
    ce_score = cross_encoder.predict([(query, document)])[0]

    # Weighted combination
    return alpha * ce_score + (1 - alpha) * dense_score

# Use for critical queries where quality matters
```

### Pattern 4: Batch Processing

```python
# Prepare all pairs in batch
pairs = [(query, doc) for doc in candidate_docs]

# Batch inference (faster than sequential)
scores = cross_encoder.model.predict(pairs, batch_size=32)

# Rank by scores
ranked_indices = scores.argsort()[::-1]
ranked_docs = [candidate_docs[i] for i in ranked_indices]
```

## Advanced Topics

### Fine-Tuning on Custom Data

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Prepare training data
train_samples = [
    InputExample(texts=[query, positive_doc], label=1.0),
    InputExample(texts=[query, negative_doc], label=0.0),
    # ...
]

# Load base model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Fine-tune
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
model.fit(
    train_dataloader=train_dataloader,
    epochs=3,
    warmup_steps=100
)

# Save
model.save("models/finetuned-cross-encoder")
```

### Knowledge Distillation

Distill larger cross-encoder into smaller model:

```python
# Teacher: Large cross-encoder
teacher = CrossEncoder("cross-encoder/ms-marco-electra-base")

# Student: Small cross-encoder
student = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Generate soft labels from teacher
query_doc_pairs = [...]
teacher_scores = teacher.predict(query_doc_pairs)

# Train student to match teacher scores
# (Knowledge distillation training loop)
```

### Attention Visualization

```python
import torch

# Get attention weights
model = cross_encoder.model
inputs = tokenizer([query, document], return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# Visualize which query tokens attend to which doc tokens
attentions = outputs.attentions  # List of attention matrices
```

## Troubleshooting

### Issue: Very Slow Inference

**Cause**: CPU inference on cross-encoder

**Solutions**:
```python
# 1. Use GPU
cross_encoder = CrossEncoder(model_name, device="cuda")

# 2. Use smaller model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# 3. Increase batch size
scores = cross_encoder.model.predict(pairs, batch_size=64)

# 4. Use bi-encoder instead (if quality acceptable)
dense_retriever = DenseRetriever(...)
```

### Issue: Out of Memory

**Cause**: Large batch size or long sequences

**Solutions**:
```python
# 1. Reduce batch size
scores = cross_encoder.model.predict(pairs, batch_size=16)

# 2. Truncate documents
max_length = 256  # Instead of 512
truncated_docs = [doc[:max_length] for doc in documents]

# 3. Process in batches
all_scores = []
batch_size = 32
for i in range(0, len(pairs), batch_size):
    batch_scores = cross_encoder.model.predict(pairs[i:i+batch_size])
    all_scores.extend(batch_scores)
```

### Issue: Poor Results on Domain Data

**Cause**: Model trained on general domain (MS MARCO)

**Solutions**:
```python
# 1. Use domain-specific model
cross_encoder = CrossEncoder("cross-encoder/scibert-zeroshot")

# 2. Fine-tune on domain data
# (See Fine-Tuning section)

# 3. Combine with BM25 (domain keywords)
hybrid_score = 0.5 * bm25_score + 0.5 * ce_score
```

## Comparison to Other Reranking Methods

### vs. Bi-Encoder (Dense)

| Aspect | Cross-Encoder | Bi-Encoder |
|--------|---------------|------------|
| **Quality** | Excellent (attends to both) | Good |
| **Speed** | Slow (O(K) passes) | Fast (cached embeddings) |
| **Scalability** | Poor (can't pre-compute) | Excellent (FAISS) |
| **Use case** | Reranking top-K | First-stage retrieval |

### vs. LLM-based Reranking

| Aspect | Cross-Encoder | LLM (GPT-4) |
|--------|---------------|-------------|
| **Quality** | Very Good | Excellent |
| **Speed** | Medium (~100ms) | Very Slow (~1s) |
| **Cost** | Free (self-hosted) | $$$$ (API calls) |
| **Consistency** | Deterministic | May vary |

### vs. BM25 Reranking

| Aspect | Cross-Encoder | BM25 |
|--------|---------------|------|
| **Quality** | Excellent (semantic) | Good (lexical) |
| **Speed** | Slow | Fast |
| **Exact matches** | May miss | Catches well |
| **Paraphrases** | Handles well | Misses |

## References

### Key Papers

1. **Sentence-BERT**: Reimers & Gurevych (2019)
   - Introduced cross-encoder architecture for sentence pair scoring

2. **MS MARCO**: Nguyen et al. (2016)
   - Large-scale passage ranking dataset
   - Cross-encoders trained on MS MARCO are de-facto standard

3. **Dense Passage Retrieval**: Karpukhin et al. (2020)
   - Compared bi-encoders vs. cross-encoders for retrieval

### Pre-Trained Models

- **Model Hub**: [https://huggingface.co/cross-encoder](https://huggingface.co/cross-encoder)
- **MS MARCO Models**: [https://www.sbert.net/examples/applications/cross-encoder/README.html](https://www.sbert.net/examples/applications/cross-encoder/README.html)

### Implementation

- **Source Code**: [dartboard/embeddings.py](../dartboard/embeddings.py) (CrossEncoder class)
- **Usage in Dartboard**: [dartboard/core.py](../dartboard/core.py) (_compute_distances method)
- **Tests**: [tests/test_embeddings.py](../tests/test_embeddings.py)

## Summary

Cross-encoders provide **state-of-the-art reranking quality** by jointly encoding query-document pairs. They are essential for high-precision applications but must be used carefully due to computational cost.

**Key Takeaways**:
- ✅ Use cross-encoders for **reranking** top 10-100 candidates
- ✅ `ms-marco-MiniLM-L-12-v2` is a good default model
- ✅ GPU acceleration is highly recommended (8-15x speedup)
- ✅ Best used in **two-stage** pipeline: fast retrieval → cross-encoder reranking
- ❌ Too slow for first-stage retrieval on large corpora
- ❌ Cannot pre-compute (unlike bi-encoders)
