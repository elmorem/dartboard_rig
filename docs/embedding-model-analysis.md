# Embedding Model Analysis & Recommendations

## Current State

### Current Default: all-MiniLM-L6-v2

**Specifications:**
- Dimensions: 384
- Model size: 80MB
- Speed: ~14K sentences/sec (CPU), ~50K sentences/sec (GPU)
- Architecture: DistilRoBERTa (6 layers)
- Pooling: Mean pooling
- Training: 1B+ sentence pairs

**Where it's used in the codebase:**
```bash
# Hardcoded default dimension
dartboard/api/dependencies.py:         EMBEDDING_DIM = 384
dartboard/datasets/synthetic.py:       embedding_dim: int = 384

# Active usage
test_scalability.py
demo_dartboard_evaluation.py
test_diversity.py
test_chunking_pipeline_integration.py
demo_chunking_endtoend.py
```

**Strengths:**
- ✅ Fast: 4-5x faster than larger models
- ✅ Small: Minimal memory footprint
- ✅ CPU-friendly: Works well without GPU
- ✅ General-purpose: Good across many domains
- ✅ Battle-tested: Most popular sentence-transformers model

**Weaknesses:**
- ❌ Lower quality: ~5% worse than best models (NDCG 0.71 vs 0.74)
- ❌ Generic: Not optimized for specific domains
- ❌ Limited context: 512 token max sequence length
- ❌ Lower dimension: 384 dims may miss nuanced semantic relationships

---

## Options for Change/Supplementation

### Option 1: Upgrade to Higher Quality Model

#### A. all-mpnet-base-v2 (Balanced Upgrade)
**Specs:**
- Dimensions: 768 (2x larger)
- Model size: 420MB (5x larger)
- Speed: ~3K sent/sec CPU (4.7x slower)
- Quality: NDCG 0.74 (+0.03), STS 72.4 (+4.3 points)

**Use case:** When quality matters more than speed
- Research applications
- High-value document collections
- Complex semantic queries
- Offline batch processing

**Implementation:**
```python
model = SentenceTransformerModel("all-mpnet-base-v2")
# Update EMBEDDING_DIM to 768 in dependencies.py
```

**Pros:**
- ✅ Best general-purpose quality
- ✅ Well-tested and reliable
- ✅ Still CPU-capable
- ✅ Worth the 5x size increase

**Cons:**
- ❌ 4.7x slower on CPU
- ❌ 2x storage requirements
- ❌ Higher memory usage

---

#### B. all-MiniLM-L12-v2 (Middle Ground)
**Specs:**
- Dimensions: 384 (same as current)
- Model size: 120MB (+50%)
- Speed: ~8K sent/sec CPU (1.75x slower)
- Quality: Between L6 and mpnet

**Use case:** Better quality without dimension change
- Same storage requirements
- Better accuracy at modest speed cost
- Drop-in replacement (384 dims)

**Implementation:**
```python
model = SentenceTransformerModel("all-MiniLM-L12-v2")
# No EMBEDDING_DIM change needed!
```

**Pros:**
- ✅ Better quality than L6-v2
- ✅ Same dimensionality (no storage increase)
- ✅ Only 1.75x slower
- ✅ Drop-in replacement

**Cons:**
- ❌ Not as fast as L6
- ❌ Not as good as mpnet

---

### Option 2: Domain-Specific Models

#### A. multi-qa-mpnet-base-dot-v1 (Question Answering)
**Specs:**
- Dimensions: 768
- Optimized for: Q&A, retrieval tasks
- Training: Trained on MS MARCO, Natural Questions, etc.

**Use case:** RAG systems focused on Q&A
- User asks questions, system retrieves answers
- Documentation search
- FAQ systems

**Benchmark (MS MARCO):**
- MRR@10: 0.347 (vs 0.328 for all-mpnet-base-v2)
- Better at matching questions to relevant passages

**Implementation:**
```python
model = SentenceTransformerModel("multi-qa-mpnet-base-dot-v1")
```

**Pros:**
- ✅ Optimized for RAG Q&A use case
- ✅ Better question-passage matching
- ✅ Trained on QA datasets

**Cons:**
- ❌ Larger (768 dims)
- ❌ May be worse for semantic similarity tasks

---

#### B. paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)
**Specs:**
- Dimensions: 384
- Languages: 50+ (including English, Spanish, French, German, Chinese, etc.)
- Performance: Similar to MiniLM-L12-v2 for English

**Use case:** International documents
- Multi-language corpus
- Cross-language search
- Global applications

**Implementation:**
```python
model = SentenceTransformerModel("paraphrase-multilingual-MiniLM-L12-v2")
```

**Pros:**
- ✅ 50+ languages
- ✅ Cross-language retrieval
- ✅ Same 384 dims
- ✅ Good English performance

**Cons:**
- ❌ Slightly worse than English-only models for English
- ❌ Larger model size

---

#### C. allenai/specter (Scientific Papers)
**Specs:**
- Dimensions: 768
- Domain: Scientific publications
- Training: Citation graphs, paper abstracts

**Use case:** Research paper RAG
- Scientific literature search
- Academic knowledge bases
- Citation-aware retrieval

**Implementation:**
```python
model = SentenceTransformerModel("allenai/specter")
```

**Pros:**
- ✅ Superior for scientific text
- ✅ Understands domain terminology
- ✅ Citation-aware

**Cons:**
- ❌ Poor for general text
- ❌ Large model

---

### Option 3: Multi-Model Approach (Hybrid Embeddings)

#### Strategy: Use different models for different purposes

**Example Architecture:**
```python
class HybridEmbeddingModel:
    def __init__(self):
        # Fast model for initial retrieval (top-100)
        self.fast_model = SentenceTransformerModel("all-MiniLM-L6-v2")

        # High-quality model for reranking (top-10)
        self.quality_model = SentenceTransformerModel("all-mpnet-base-v2")

    def retrieve(self, query, k=10):
        # Stage 1: Fast retrieval with L6 model
        fast_embedding = self.fast_model.encode(query)
        candidates = vector_store.search(fast_embedding, k=100)

        # Stage 2: Rerank top-100 with mpnet
        query_emb = self.quality_model.encode(query)
        candidate_embs = self.quality_model.encode([c.text for c in candidates])

        # Compute similarities and rerank
        scores = [np.dot(query_emb, emb) for emb in candidate_embs]
        top_k = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]

        return [c for c, _ in top_k]
```

**Pros:**
- ✅ Best of both worlds: speed + quality
- ✅ Fast initial retrieval
- ✅ High-quality final ranking
- ✅ Flexible per-use-case

**Cons:**
- ❌ Complexity: Two models to manage
- ❌ Storage: Need both models in memory
- ❌ Implementation: More code

---

### Option 4: Fine-Tuned Model

#### When to fine-tune:
- ✅ Domain-specific vocabulary (medical, legal, financial)
- ✅ Poor performance with pre-trained models (<60% accuracy)
- ✅ Have 1000+ labeled query-document pairs

#### Process:
```python
from sentence_transformers import SentenceTransformer, InputExample, losses

# 1. Collect training data (your RAG logs!)
train_examples = [
    InputExample(texts=["user query", "relevant chunk"], label=1.0),
    InputExample(texts=["user query", "irrelevant chunk"], label=0.0),
    # ... 1000+ examples from actual usage
]

# 2. Fine-tune base model
model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(dataloader, train_loss)],
    epochs=3,
    output_path="models/dartboard-finetuned"
)
```

**Pros:**
- ✅ Best quality for your specific domain
- ✅ Learns from actual usage patterns
- ✅ Can start with small model and improve it

**Cons:**
- ❌ Requires GPU for training
- ❌ Needs quality training data
- ❌ Ongoing maintenance

---

### Option 5: Latest State-of-the-Art Models

#### A. intfloat/e5-small-v2 (2023)
**Specs:**
- Dimensions: 384
- Model: E5 (Text Embeddings by Weakly-Supervised Contrastive Pre-training)
- Performance: Better than all-MiniLM on BEIR benchmark

**Benchmarks (BEIR avg):**
- e5-small-v2: 0.467
- all-MiniLM-L6-v2: 0.420
- Improvement: +11% accuracy

**Implementation:**
```python
model = SentenceTransformerModel("intfloat/e5-small-v2")
```

**Pros:**
- ✅ State-of-the-art for 384 dims
- ✅ Same size as current model
- ✅ Better retrieval quality
- ✅ Drop-in replacement

**Cons:**
- ❌ Newer, less battle-tested
- ❌ May be slightly slower

---

#### B. BAAI/bge-small-en-v1.5 (2023)
**Specs:**
- Dimensions: 384
- Model: BGE (BAAI General Embedding)
- Performance: SOTA on MTEB benchmark

**Benchmarks (MTEB):**
- bge-small-en-v1.5: 0.620
- all-MiniLM-L6-v2: 0.583
- Improvement: +6% accuracy

**Implementation:**
```python
model = SentenceTransformerModel("BAAI/bge-small-en-v1.5")
```

**Pros:**
- ✅ SOTA performance
- ✅ Same 384 dims
- ✅ Well-documented
- ✅ Active development

**Cons:**
- ❌ Newer model
- ❌ Less community usage

---

## Recommendations

### Recommendation 1: **Immediate Upgrade (Low Risk)**
**Switch to: all-MiniLM-L12-v2**

**Why:**
- Same 384 dimensions (no storage changes)
- Better quality at modest speed cost (1.75x slower)
- Drop-in replacement
- Low risk, clear improvement

**Implementation:**
```python
# dartboard/api/dependencies.py
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")

# Users can override with environment variable
# EMBEDDING_MODEL=all-mpnet-base-v2 for high quality
# EMBEDDING_MODEL=all-MiniLM-L6-v2 for max speed
```

---

### Recommendation 2: **For Quality-Focused Applications**
**Switch to: all-mpnet-base-v2**

**Why:**
- Best general-purpose model
- Worth the 2x storage for 5% quality gain
- Still CPU-capable
- Production-proven

**Tradeoffs:**
- 2x storage (768 vs 384 dims)
- 4.7x slower on CPU (use GPU)

**Implementation:**
```python
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
```

---

### Recommendation 3: **Modern Alternative (Experimental)**
**Try: intfloat/e5-small-v2**

**Why:**
- State-of-the-art 384-dim model
- +11% retrieval accuracy over MiniLM-L6
- Same storage requirements
- Modern architecture

**Risk:**
- Newer, less tested
- May have compatibility issues

**Implementation:**
```python
EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIM = 384
```

---

### Recommendation 4: **Two-Stage Hybrid (Advanced)**
**Use: MiniLM-L6 for retrieval + mpnet for reranking**

**Why:**
- Best balance of speed and quality
- Fast initial search (14K sent/sec)
- High-quality final results
- Minimal latency impact (only rerank top-10)

**Implementation:**
See "Option 3: Multi-Model Approach" above

---

## Implementation Plan

### Phase 1: Make Model Configurable (Priority 1)
```python
# dartboard/api/dependencies.py
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

def get_embedding_model():
    return SentenceTransformerModel(EMBEDDING_MODEL_NAME)
```

**Benefits:**
- Users can experiment with different models
- No code changes needed
- Easy A/B testing

---

### Phase 2: Add Model Comparison Tool (Priority 2)
```python
# dartboard/evaluation/embedding_comparison.py

class EmbeddingModelComparator:
    def compare_models(self, models: List[str], queries: List[str],
                       ground_truth: List[List[str]]):
        """
        Compare retrieval quality across embedding models.

        Returns metrics: MRR, NDCG, Recall@K for each model
        """
        results = {}
        for model_name in models:
            model = SentenceTransformerModel(model_name)
            # Run retrieval evaluation
            metrics = self.evaluate(model, queries, ground_truth)
            results[model_name] = metrics
        return results
```

**Benefits:**
- Data-driven model selection
- Benchmark on your actual data
- Compare speed vs quality tradeoffs

---

### Phase 3: Implement Hybrid Retrieval (Priority 3)
```python
# dartboard/retrieval/hybrid_embedding.py

class HybridEmbeddingRetriever:
    def __init__(self, fast_model, quality_model):
        self.fast_model = fast_model  # L6-v2
        self.quality_model = quality_model  # mpnet

    def retrieve(self, query, k=10):
        # Stage 1: Fast retrieval (top-100)
        candidates = self._fast_retrieve(query, k=100)

        # Stage 2: Quality reranking (top-10)
        results = self._rerank(query, candidates, k=k)

        return results
```

**Benefits:**
- Best of both worlds
- Minimal latency increase
- Maximum quality

---

## Evaluation Metrics

### How to choose the right model:

1. **Run on your data**:
```python
from dartboard.evaluation.metrics import RetrievalMetrics

# Test on your actual queries
metrics = RetrievalMetrics()
results = metrics.evaluate(
    queries=your_queries,
    ground_truth=your_relevant_docs,
    model="all-MiniLM-L6-v2"
)

print(f"NDCG@10: {results.ndcg_at_10}")
print(f"MRR: {results.mrr}")
print(f"Recall@10: {results.recall_at_10}")
```

2. **Compare models side-by-side**:
```python
models_to_test = [
    "all-MiniLM-L6-v2",      # Current default
    "all-MiniLM-L12-v2",     # Better quality
    "all-mpnet-base-v2",     # Best quality
    "intfloat/e5-small-v2"   # Modern SOTA
]

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")
    results = metrics.evaluate(queries, ground_truth, model_name)
    print(results)
```

3. **Measure speed**:
```python
import time

texts = [chunk.text for chunk in chunks]

start = time.time()
embeddings = model.encode(texts, batch_size=32)
elapsed = time.time() - start

print(f"Speed: {len(texts) / elapsed:.0f} sentences/sec")
```

---

## Summary Table

| Model | Dims | Speed (CPU) | Quality | Use Case | Risk |
|-------|------|-------------|---------|----------|------|
| all-MiniLM-L6-v2 | 384 | 14K/sec | Good | **Current default** | ✅ Low |
| all-MiniLM-L12-v2 | 384 | 8K/sec | Better | **Recommended upgrade** | ✅ Low |
| all-mpnet-base-v2 | 768 | 3K/sec | Best | High-quality RAG | ✅ Low |
| intfloat/e5-small-v2 | 384 | 10K/sec | Excellent | Modern alternative | ⚠️ Medium |
| BAAI/bge-small-en-v1.5 | 384 | 10K/sec | Excellent | SOTA performance | ⚠️ Medium |
| multi-qa-mpnet | 768 | 3K/sec | Best (Q&A) | RAG Q&A systems | ✅ Low |
| Hybrid (L6+mpnet) | Both | Fast+Slow | Best | Production RAG | ⚠️ High |

---

## Next Steps

1. **Immediate**: Make embedding model configurable via environment variables
2. **Short-term**: Test all-MiniLM-L12-v2 as new default
3. **Medium-term**: Build model comparison tool for your data
4. **Long-term**: Consider fine-tuning or hybrid approach

**Question for you:** What's your primary concern?
- 🎯 **Quality**: Go with all-mpnet-base-v2 or multi-qa-mpnet
- ⚡ **Speed**: Stay with all-MiniLM-L6-v2 or try e5-small-v2
- 💰 **Balance**: Switch to all-MiniLM-L12-v2 (recommended)
- 🚀 **Cutting-edge**: Try intfloat/e5-small-v2 or BAAI/bge
