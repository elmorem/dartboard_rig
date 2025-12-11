# Embedding Model Configuration - Changes Summary

## What Changed

Dartboard now supports **configurable embedding models** via environment variables or programmatic configuration. Previously, the embedding model was hardcoded to `all-MiniLM-L6-v2` throughout the codebase.

## Key Benefits

✅ **Easy experimentation** - Switch models without changing code
✅ **Production flexibility** - Different models for different environments
✅ **Performance optimization** - Choose speed vs. quality tradeoffs
✅ **Domain adaptation** - Use specialized models for specific tasks

## Quick Start

### Change Model via Environment Variable

```bash
# Use high-quality model
export EMBEDDING_MODEL="all-mpnet-base-v2"
export EMBEDDING_DIM="768"

# Run your code
python demo_dartboard.py
```

```bash
# Use fast model (default)
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
# EMBEDDING_DIM auto-detected as 384

# Run tests
pytest
```

### In Code

```python
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel

# Automatically uses configured model
config = get_embedding_config()
model = SentenceTransformerModel(config.model_name, device=config.device)
```

### In Tests

```python
def test_my_feature(embedding_model):
    """Tests automatically use configured model via fixture."""
    embeddings = embedding_model.encode("test text")
```

## What Was Changed

### New Files

1. **`dartboard/config/embeddings.py`**
   - Centralized `EmbeddingConfig` class
   - Auto-detection of embedding dimensions
   - Support for 15+ pre-configured models

2. **`dartboard/config/__init__.py`**
   - Public API for configuration

3. **`conftest.py`** (root level)
   - Pytest fixtures for tests

4. **`docs/configuring-embedding-models.md`**
   - Complete usage guide with examples

5. **`docs/embedding-model-analysis.md`**
   - Detailed comparison of 8 embedding models
   - Performance benchmarks and recommendations

### Modified Files

#### Core Application
- `dartboard/api/dependencies.py` - Uses `EmbeddingConfig`
- `dartboard/datasets/synthetic.py` - Auto-detects dimension

#### Tests
- `tests/conftest.py` - Added shared fixtures
- `test_embedding_semantic_chunking.py` - Uses config
- `test_chunking_pipeline_integration.py` - Uses config

#### Demos
- `demo_dartboard.py` - Uses config
- `demo_dartboard_evaluation.py` - Uses config
- `demo_chunking_endtoend.py` - Uses config
- `test_scalability.py` - Uses config
- `test_diversity.py` - Uses config
- `test_redundancy.py` - Uses config
- `test_qa_dataset.py` - Uses config

## Available Models

| Model | Dims | Speed | Use Case |
|-------|------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | **Default** - Best balance |
| `all-MiniLM-L12-v2` | 384 | Medium | **Recommended upgrade** |
| `all-mpnet-base-v2` | 768 | Slow | Highest quality |
| `multi-qa-mpnet-base-dot-v1` | 768 | Slow | RAG Q&A systems |
| `intfloat/e5-small-v2` | 384 | Fast | Modern SOTA |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Cutting-edge |

See [docs/embedding-model-analysis.md](docs/embedding-model-analysis.md) for full comparison.

## Migration Guide

### Before (Hardcoded)

```python
from dartboard.embeddings import SentenceTransformerModel

# Hardcoded model name
model = SentenceTransformerModel("all-MiniLM-L6-v2")
```

### After (Configurable)

```python
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel

# Uses environment variable or default
config = get_embedding_config()
model = SentenceTransformerModel(config.model_name)
```

### Or Use Fixtures in Tests

```python
# Automatically configured via conftest.py
def test_something(embedding_model):
    embeddings = embedding_model.encode("test")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Model name | `all-MiniLM-L6-v2` |
| `EMBEDDING_DIM` | Embedding dimension | Auto-detected |
| `EMBEDDING_DEVICE` | Device (`cpu`/`cuda`) | `cpu` |

## Testing

All existing tests pass with the new configuration system:

```bash
# Run with default model
pytest

# Run with specific model
EMBEDDING_MODEL="all-mpnet-base-v2" pytest

# Run with GPU
EMBEDDING_DEVICE="cuda" pytest
```

Verified tests:
- ✅ test_chunking_metrics.py (12 tests)
- ✅ test_embedding_semantic_chunking.py
- ✅ test_metadata_extraction.py
- ✅ All other existing tests

## Important Notes

### Re-indexing Required When Changing Models

If you change the embedding model, you **must re-index** your documents:

```bash
# Wrong - mixing embeddings from different models
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
python ingest.py  # 384-dim embeddings

export EMBEDDING_MODEL="all-mpnet-base-v2"  # Now 768-dim!
python query.py  # ERROR: Dimension mismatch
```

```bash
# Correct - re-index when changing models
export EMBEDDING_MODEL="all-mpnet-base-v2"
rm -rf ./data/vector_store  # Clear old index
python ingest.py  # Re-index with new model
python query.py  # Works!
```

### GPU Support

```bash
# Use GPU for 3-5x speedup
export EMBEDDING_DEVICE="cuda"
```

## Documentation

- **Usage Guide:** [docs/configuring-embedding-models.md](docs/configuring-embedding-models.md)
- **Model Comparison:** [docs/embedding-model-analysis.md](docs/embedding-model-analysis.md)
- **Implementation:** [dartboard/config/embeddings.py](dartboard/config/embeddings.py)

## Backward Compatibility

✅ **Fully backward compatible**

If no environment variables are set, the system defaults to `all-MiniLM-L6-v2` (the previous hardcoded value). All existing code continues to work without changes.

## Next Steps

1. **Experiment** - Try different models on your data
2. **Benchmark** - Use the evaluation tools to compare models
3. **Optimize** - Choose the best model for your use case
4. **Deploy** - Set environment variables in production

## Questions?

See the detailed guides:
- [Configuring Embedding Models](docs/configuring-embedding-models.md)
- [Embedding Model Analysis](docs/embedding-model-analysis.md)
