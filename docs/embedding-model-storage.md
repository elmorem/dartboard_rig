# Embedding Model Storage & Access

## Overview

Embedding models in Dartboard are accessed through the **sentence-transformers** library, which automatically downloads pre-trained models from **HuggingFace Hub** and caches them locally.

## How It Works

### 1. The Library: sentence-transformers

```python
# In dartboard/embeddings.py (line 49)
from sentence_transformers import SentenceTransformer

self.model = SentenceTransformer(model_name, device=device)
```
                                                          
**What it is:**
- Python library built on top of HuggingFace Transformers
- Optimized specifically for sentence/document embeddings
- Handles all the complexity: download, caching, tokenization, inference

**Key features:**
- Automatic model download and caching
- Batch processing for efficiency
- GPU support
- 100+ pre-trained models available

### 2. The Source: HuggingFace Hub

**What it is:**
- Public repository of machine learning models
- Free to access and use (Apache 2.0 / MIT licenses)
- 100,000+ models (transformers, diffusion, etc.)
- Managed by HuggingFace (https://huggingface.co)

**Example model URLs:**
- `all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- `all-mpnet-base-v2`: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- `intfloat/e5-small-v2`: https://huggingface.co/intfloat/e5-small-v2

### 3. Local Storage: Automatic Caching

**Default cache location:**
```bash
~/.cache/torch/sentence_transformers/
```

**On macOS/Linux:**
```
/Users/yourusername/.cache/torch/sentence_transformers/
  └── sentence-transformers_all-MiniLM-L6-v2/
      ├── config.json
      ├── pytorch_model.bin
      ├── tokenizer.json
      ├── vocab.txt
      └── ... (other files)
```

**On Windows:**
```
C:\Users\yourusername\.cache\torch\sentence_transformers\
```

## Download Process

### First Time You Use a Model

```python
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel

# First time loading all-MiniLM-L6-v2
model = SentenceTransformerModel("all-MiniLM-L6-v2")
```

**What happens:**

1. **Check cache:** sentence-transformers looks in `~/.cache/torch/sentence_transformers/`
2. **Not found:** Initiates download from HuggingFace Hub
3. **Download:** Pulls model files (~80MB for all-MiniLM-L6-v2)
   ```
   Downloading model from HuggingFace Hub...
   all-MiniLM-L6-v2: 100%|████████| 90.9M/90.9M [00:15<00:00, 6.05MB/s]
   ```
4. **Save to cache:** Stores files locally
5. **Load into memory:** Model ready to use

### Subsequent Uses

```python
# Second time and onwards
model = SentenceTransformerModel("all-MiniLM-L6-v2")  # Instant load from cache!
```

**What happens:**

1. **Check cache:** Found in `~/.cache/`
2. **Load directly:** No download needed
3. **Ready instantly:** Typically <1 second

## Model File Structure

When you download a model, you get:

```
sentence-transformers_all-MiniLM-L6-v2/
├── config.json              # Model architecture config
├── pytorch_model.bin        # Pre-trained weights (80MB)
├── tokenizer_config.json    # Tokenizer settings
├── tokenizer.json           # Vocabulary and tokenization rules
├── vocab.txt                # Word piece vocabulary
├── special_tokens_map.json  # Special tokens ([CLS], [SEP], etc.)
├── modules.json             # Layer configurations
└── README.md                # Model card with details
```

**Total size examples:**
- `all-MiniLM-L6-v2`: ~80-90 MB
- `all-MiniLM-L12-v2`: ~120 MB
- `all-mpnet-base-v2`: ~420 MB

## How Dartboard Accesses Models

### Layer 1: Your Code

```python
# Your application or test
from dartboard.config import get_embedding_config

config = get_embedding_config()  # Reads EMBEDDING_MODEL env var
```

### Layer 2: Dartboard Config

```python
# dartboard/config/embeddings.py
class EmbeddingConfig:
    def __init__(self, model_name=None, ...):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
```

### Layer 3: Dartboard Wrapper

```python
# dartboard/embeddings.py
class SentenceTransformerModel:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
```

### Layer 4: sentence-transformers Library

```python
# Inside sentence-transformers (hidden from you)
class SentenceTransformer:
    def __init__(self, model_name_or_path):
        # 1. Check cache
        # 2. Download if needed
        # 3. Load PyTorch model
        # 4. Load tokenizer
```

### Layer 5: HuggingFace Hub

```
HTTPS request to: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Downloads: pytorch_model.bin, config.json, tokenizer.json, etc.
```

## Controlling Model Storage

### Change Cache Directory

```bash
# Set custom cache location
export SENTENCE_TRANSFORMERS_HOME="/path/to/custom/cache"

# Or in Python
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/path/to/cache'

# Then load model
from dartboard.embeddings import SentenceTransformerModel
model = SentenceTransformerModel("all-MiniLM-L6-v2")
```

### Pre-download Models

```python
# Download models ahead of time (e.g., in Docker build)
from sentence_transformers import SentenceTransformer

# Download common models
models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1"
]

for model_name in models:
    print(f"Downloading {model_name}...")
    SentenceTransformer(model_name)
    print(f"✓ {model_name} cached")
```

### Check What's Cached

```bash
# List cached models
ls -lh ~/.cache/torch/sentence_transformers/

# Check cache size
du -sh ~/.cache/torch/sentence_transformers/
```

## Offline Usage

### Option 1: Use Pre-downloaded Models

```bash
# Download on a machine with internet
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy cache to offline machine
scp -r ~/.cache/torch/sentence_transformers/ user@offline-machine:~/.cache/torch/

# On offline machine, models load from cache
python demo_dartboard.py  # Works without internet!
```

### Option 2: Use Local Model Files

```bash
# Download model files manually from HuggingFace
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
# ... download all files

# Load from local directory
export EMBEDDING_MODEL="/path/to/local/model/directory"
python demo_dartboard.py
```

## Docker Deployment

### Dockerfile Example

```dockerfile
FROM python:3.13-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download embedding models during build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    SentenceTransformer('all-mpnet-base-v2')"

# Copy application code
COPY . /app
WORKDIR /app

# Models are now baked into the image
CMD ["python", "demo_dartboard.py"]
```

**Benefits:**
- Models downloaded once during build
- Image size increases by model size (~80-420MB per model)
- No download needed at runtime
- Faster container startup

### Docker Compose with Volume

```yaml
# docker-compose.yml
services:
  dartboard:
    image: dartboard-rag
    volumes:
      - ./data:/app/data
      - model-cache:/root/.cache/torch/sentence_transformers
    environment:
      - EMBEDDING_MODEL=all-MiniLM-L6-v2

volumes:
  model-cache:  # Persist downloaded models
```

## Custom/Fine-tuned Models

### Using a Local Fine-tuned Model

```python
# Save your fine-tuned model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# ... fine-tune the model ...
model.save('/path/to/my-finetuned-model')
```

```bash
# Use it in Dartboard
export EMBEDDING_MODEL="/path/to/my-finetuned-model"
export EMBEDDING_DIM="384"  # Specify dimension manually

python demo_dartboard.py
```

### Sharing Custom Models

You can upload your fine-tuned model to HuggingFace Hub:

```python
# Upload to HuggingFace Hub
model.save_to_hub("myusername/my-custom-embedding-model")
```

```bash
# Use it in Dartboard
export EMBEDDING_MODEL="myusername/my-custom-embedding-model"
python demo_dartboard.py
```

## Monitoring Model Usage

### Check Which Model is Loaded

```python
from dartboard.config import get_embedding_config

config = get_embedding_config()
print(f"Active model: {config.model_name}")
print(f"Dimension: {config.embedding_dim}")
print(f"Device: {config.device}")
```

### Log Model Loading

```python
# In dartboard/api/dependencies.py (already implemented)
logger.info(
    f"Initializing embedding model: {EMBEDDING_CONFIG.model_name} "
    f"(dim={EMBEDDING_CONFIG.embedding_dim}, device={EMBEDDING_CONFIG.device})"
)
```

## Troubleshooting

### Issue: Download Fails

```
URLError: <urlopen error [Errno 8] nodename nor servname provided, or not known>
```

**Solutions:**

1. **Check internet connection**
2. **Use proxy if needed:**
   ```bash
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```
3. **Download manually** and load from local path

### Issue: Cache Corruption

```
RuntimeError: Error(s) in loading state_dict
```

**Solution:** Clear cache and re-download

```bash
rm -rf ~/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: Disk Space

```
OSError: [Errno 28] No space left on device
```

**Solution:** Clear old models or move cache

```bash
# Check cache size
du -sh ~/.cache/torch/sentence_transformers/

# Remove specific model
rm -rf ~/.cache/torch/sentence_transformers/sentence-transformers_old-model

# Move cache to larger disk
mv ~/.cache/torch/sentence_transformers /mnt/large-disk/model-cache
ln -s /mnt/large-disk/model-cache ~/.cache/torch/sentence_transformers
```

## Summary

**How it works:**

1. **You specify:** `EMBEDDING_MODEL="all-MiniLM-L6-v2"`
2. **Dartboard wraps:** `SentenceTransformerModel(config.model_name)`
3. **sentence-transformers downloads:** From HuggingFace Hub (first time)
4. **Models cached:** In `~/.cache/torch/sentence_transformers/`
5. **Subsequent loads:** Instant (from cache)

**Key points:**

- ✅ Models downloaded automatically from HuggingFace Hub
- ✅ Cached locally for fast subsequent loads
- ✅ No manual download needed
- ✅ Free and open source models
- ✅ Works offline after first download
- ✅ Docker-friendly (pre-download in image build)

**Model locations:**

- **Online source:** https://huggingface.co/sentence-transformers/
- **Local cache:** `~/.cache/torch/sentence_transformers/`
- **Custom models:** Any local path or HuggingFace Hub repo

For more information:
- [HuggingFace Hub](https://huggingface.co/)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [Configuring Embedding Models](./configuring-embedding-models.md)
