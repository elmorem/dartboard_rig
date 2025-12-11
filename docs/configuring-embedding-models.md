# Configuring Embedding Models

## Overview

Dartboard now supports configurable embedding models, allowing you to easily switch between different models without changing code. This enables you to:

- Experiment with different embedding models
- Optimize for speed vs. quality tradeoffs
- Use domain-specific models for specialized tasks
- Test cutting-edge models as they become available

> **Note:** Models are automatically downloaded from HuggingFace Hub and cached locally. See [Embedding Model Storage](./embedding-model-storage.md) for details on where models are stored and how they're accessed.

## Quick Start

### Using Environment Variables

The simplest way to configure the embedding model is via environment variables:

```bash
# Use a high-quality model
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"

# Run your application
python demo_dartboard.py
```

```bash
# Use a fast, efficient model (default)
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_DIM="384"

# Run tests
pytest
```

### Supported Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Model name (HuggingFace model ID) | `all-MiniLM-L6-v2` |
| `EMBEDDING_DIM` | Embedding dimension | Auto-detected from model |
| `EMBEDDING_DEVICE` | Device to run on (`cpu` or `cuda`) | `cpu` |

## Available Models

### General Purpose Models

#### all-MiniLM-L6-v2 (Default - Best Balance)
```bash
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
# EMBEDDING_DIM=384 (auto-detected)
```
- **Dimensions:** 384
- **Speed:** ~14K sentences/sec (CPU)
- **Use case:** Default for most applications
- **Pros:** Fast, efficient, good quality
- **Cons:** Lower quality than larger models

#### all-MiniLM-L12-v2 (Recommended Upgrade)
```bash
export EMBEDDING_MODEL="all-MiniLM-L12-v2"
# EMBEDDING_DIM=384 (auto-detected)
```
- **Dimensions:** 384
- **Speed:** ~8K sentences/sec (CPU)
- **Use case:** Better quality without storage increase
- **Pros:** Same dims as L6, better accuracy
- **Cons:** 1.75x slower than L6

#### all-mpnet-base-v2 (Highest Quality)
```bash
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"
```
- **Dimensions:** 768
- **Speed:** ~3K sentences/sec (CPU)
- **Use case:** When quality is paramount
- **Pros:** Best general-purpose model
- **Cons:** 2x storage, 4.7x slower

### Domain-Specific Models

#### multi-qa-mpnet-base-dot-v1 (Question Answering)
```bash
export EMBEDDING_MODEL="multi-qa-mpnet-base-dot-v1"
export EMBEDDING_DIM="768"
```
- **Optimized for:** RAG Q&A systems
- **Training:** MS MARCO, Natural Questions
- **Use case:** When users ask questions

#### paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)
```bash
export EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_DIM=384 (auto-detected)
```
- **Languages:** 50+
- **Use case:** International documents
- **Pros:** Cross-language retrieval

### State-of-the-Art Models (2023)

#### intfloat/e5-small-v2 (Modern, Efficient)
```bash
export EMBEDDING_MODEL="intfloat/e5-small-v2"
# EMBEDDING_DIM=384 (auto-detected)
```
- **Performance:** +11% over MiniLM-L6 on BEIR
- **Use case:** Modern alternative to MiniLM
- **Note:** Newer model, less battle-tested

#### BAAI/bge-small-en-v1.5 (SOTA)
```bash
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
# EMBEDDING_DIM=384 (auto-detected)
```
- **Performance:** SOTA on MTEB benchmark
- **Use case:** Cutting-edge performance
- **Note:** Active development

## Programmatic Configuration

### In Application Code

```python
from dartboard.config import EmbeddingConfig, get_embedding_config
from dartboard.embeddings import SentenceTransformerModel

# Option 1: Use default config (reads from environment)
config = get_embedding_config()
model = SentenceTransformerModel(config.model_name, device=config.device)

# Option 2: Override programmatically
from dartboard.config import EmbeddingConfig

config = EmbeddingConfig(
    model_name="all-mpnet-base-v2",
    device="cuda",
    embedding_dim=768
)

model = SentenceTransformerModel(config.model_name, device=config.device)
```

### In Tests

Tests automatically use the configured embedding model via pytest fixtures:

```python
def test_my_feature(embedding_model, embedding_config):
    """Test using configured embedding model."""
    # embedding_model is automatically provided based on configuration
    embeddings = embedding_model.encode("test text")

    # embedding_config provides access to settings
    assert embedding_config.embedding_dim in [384, 768]
```

To test with a specific model:

```bash
# Run tests with high-quality model
EMBEDDING_MODEL="all-mpnet-base-v2" pytest

# Run tests with fast model
EMBEDDING_MODEL="all-MiniLM-L6-v2" pytest
```

### In API/FastAPI Applications

The API automatically uses the configured model:

```bash
# Start API with specific model
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"
uvicorn dartboard.api.main:app
```

Check current configuration:

```bash
curl http://localhost:8000/config
```

Response:
```json
{
  "embedding_model": "all-mpnet-base-v2",
  "embedding_dim": 768,
  "embedding_device": "cpu",
  "vector_store_path": "./data/vector_store",
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo"
}
```

## Migration Guide

### Before (Hardcoded)

```python
# Old way - hardcoded model
from dartboard.embeddings import SentenceTransformerModel

model = SentenceTransformerModel("all-MiniLM-L6-v2")
```

### After (Configurable)

```python
# New way - configurable
from dartboard.embeddings import SentenceTransformerModel
from dartboard.config import get_embedding_config

config = get_embedding_config()
model = SentenceTransformerModel(config.model_name, device=config.device)
```

Or use the pytest fixture in tests:

```python
# Even simpler in tests
def test_something(embedding_model):
    # embedding_model automatically configured
    embeddings = embedding_model.encode("text")
```

## Comparing Models

### Benchmark on Your Data

```python
from dartboard.config import EmbeddingConfig
from dartboard.embeddings import SentenceTransformerModel
from dartboard.evaluation.metrics import RetrievalMetrics

models_to_test = [
    "all-MiniLM-L6-v2",      # Fast
    "all-MiniLM-L12-v2",     # Balanced
    "all-mpnet-base-v2",     # Quality
    "intfloat/e5-small-v2"   # Modern
]

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")

    config = EmbeddingConfig(model_name=model_name)
    model = SentenceTransformerModel(config.model_name)

    # Run your evaluation
    metrics = evaluate_retrieval(model, your_queries, your_docs)
    print(f"NDCG@10: {metrics.ndcg_at_10:.3f}")
    print(f"MRR: {metrics.mrr:.3f}")
```

### Speed Benchmark

```python
import time
from dartboard.config import EmbeddingConfig
from dartboard.embeddings import SentenceTransformerModel

def benchmark_model(model_name, texts):
    """Benchmark encoding speed."""
    config = EmbeddingConfig(model_name=model_name)
    model = SentenceTransformerModel(config.model_name)

    start = time.time()
    embeddings = model.encode(texts, batch_size=32)
    elapsed = time.time() - start

    speed = len(texts) / elapsed
    print(f"{model_name}: {speed:.0f} sentences/sec")
    return speed

# Test models
texts = ["Sample text"] * 1000
benchmark_model("all-MiniLM-L6-v2", texts)
benchmark_model("all-mpnet-base-v2", texts)
```

## Best Practices

### 1. Choose the Right Model for Your Use Case

**For Speed (Real-time, High Volume):**
```bash
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

**For Quality (Research, High-Value Docs):**
```bash
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"
```

**For Balance (Recommended):**
```bash
export EMBEDDING_MODEL="all-MiniLM-L12-v2"
```

**For Q&A RAG:**
```bash
export EMBEDDING_MODEL="multi-qa-mpnet-base-dot-v1"
export EMBEDDING_DIM="768"
```

### 2. Use GPU When Available

```bash
export EMBEDDING_DEVICE="cuda"
```

Speed increase:
- MiniLM-L6: 14K → 50K sentences/sec (3.5x faster)
- mpnet-base: 3K → 15K sentences/sec (5x faster)

### 3. Maintain Consistency

**Important:** If you change embedding models, you must re-index all documents. Embeddings from different models are not compatible.

```bash
# Wrong: Mixing embeddings from different models
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
python ingest_documents.py  # Creates 384-dim embeddings

export EMBEDDING_MODEL="all-mpnet-base-v2"  # Now 768-dim!
python query.py  # ERROR: Dimension mismatch!
```

```bash
# Correct: Re-index when changing models
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"
rm -rf ./data/vector_store  # Clear old embeddings
python ingest_documents.py  # Re-index with new model
python query.py  # Now works!
```

### 4. Test Before Production

Always benchmark on your actual data:

```bash
# Test with different models
EMBEDDING_MODEL="all-MiniLM-L6-v2" python evaluate.py > results_l6.txt
EMBEDDING_MODEL="all-MiniLM-L12-v2" python evaluate.py > results_l12.txt
EMBEDDING_MODEL="all-mpnet-base-v2" python evaluate.py > results_mpnet.txt

# Compare results
diff results_l6.txt results_l12.txt
```

## Troubleshooting

### Issue: Dimension Mismatch

```
ValueError: Embedding dimension mismatch: expected 384, got 768
```

**Solution:** Clear vector store and re-index:

```bash
rm -rf ./data/vector_store
export EMBEDDING_DIM="768"  # Match your model
python ingest_documents.py
```

### Issue: Model Not Found

```
OSError: [Model 'my-custom-model' not found]
```

**Solution:** Verify model name on HuggingFace:

```bash
# Check model exists
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

### Issue: Out of Memory

```
CUDA out of memory
```

**Solutions:**

```bash
# 1. Use CPU
export EMBEDDING_DEVICE="cpu"

# 2. Use smaller model
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# 3. Reduce batch size in code
embeddings = model.encode(texts, batch_size=16)  # Default is 32
```

## Advanced: Custom Models

### Using a Fine-Tuned Model

```bash
# Local fine-tuned model
export EMBEDDING_MODEL="/path/to/my/finetuned-model"
export EMBEDDING_DIM="384"  # Specify dimension
```

### Registering New Models

Add to `dartboard/config/embeddings.py`:

```python
MODEL_DIMENSIONS = {
    # ... existing models ...
    "my-custom-model": 512,  # Add your custom model
}
```

## Summary

✅ **Set environment variables** to change models
✅ **Use `get_embedding_config()`** in application code
✅ **Use pytest fixtures** in tests
✅ **Benchmark on your data** before switching
✅ **Re-index when changing models**
✅ **Use GPU** for production deployments

For more details, see:

- [Embedding Model Storage](./embedding-model-storage.md) - Where models are stored and how they're accessed
- [Embedding Model Analysis](./embedding-model-analysis.md) - Detailed comparison of models
- [Embeddings Documentation](./embeddings.md) - Technical deep dive
- [Source Code](../dartboard/config/embeddings.py) - Configuration implementation
