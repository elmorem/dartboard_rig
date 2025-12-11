# Embeddings

## Overview

**Embeddings** are dense vector representations of text that capture semantic meaning in a high-dimensional space. In this RAG system, embeddings power semantic search by enabling similarity comparisons between queries and documents.

## Why Embeddings Matter for RAG

1. **Semantic Search**: Match queries to documents by meaning, not just keywords
2. **Dense Vectors**: Fixed-size representations (384-1024 dims) regardless of text length
3. **Similarity Computation**: Cosine similarity measures semantic relatedness
4. **Transfer Learning**: Pre-trained models work across domains

**Example**:
```
"machine learning" → [0.23, -0.15, 0.67, ..., 0.41]  (384 dimensions)
"ML algorithms"    → [0.25, -0.13, 0.69, ..., 0.39]  (similar vector!)
"banana recipe"    → [-0.87, 0.42, -0.31, ..., 0.18] (different vector)
```

## Embedding Models in This Repository

### Default Model: all-MiniLM-L6-v2

**Why this model was chosen**:

1. **Optimal Speed-Quality Trade-off**
   - 384 dimensions (smaller = faster)
   - 6 transformer layers (vs. 12 in larger models)
   - ~14,000 sentences/sec on CPU
   - 90% accuracy of larger models at 5x speed

2. **General-Purpose Performance**
   - Trained on 1B+ sentence pairs
   - Strong performance across domains
   - Good for question answering, semantic search, clustering

3. **Resource Efficiency**
   - Model size: 80MB (vs. 400MB for larger models)
   - Memory footprint: ~500MB during inference
   - CPU-friendly (GPU optional)

4. **Proven Track Record**
   - Most popular sentence-transformers model
   - Used in production by thousands of applications
   - Extensive benchmarking and validation

### Model Specifications

```python
from dartboard.embeddings import SentenceTransformerModel

model = SentenceTransformerModel("all-MiniLM-L6-v2")

# Specifications:
# - Dimensions: 384
# - Max sequence length: 512 tokens
# - Architecture: DistilRoBERTa (6 layers)
# - Pooling: Mean pooling
# - Normalized: Yes (unit vectors)
```

### Performance Benchmarks

| Metric | all-MiniLM-L6-v2 | all-mpnet-base-v2 | Notes |
|--------|------------------|-------------------|-------|
| **Embedding Dim** | 384 | 768 | 2x smaller |
| **Model Size** | 80MB | 420MB | 5x smaller |
| **CPU Speed** | 14K sent/sec | 3K sent/sec | 4.7x faster |
| **GPU Speed** | 50K sent/sec | 15K sent/sec | 3.3x faster |
| **STS Accuracy** | 68.1 | 72.4 | -4.3 points |
| **Retrieval NDCG** | 0.71 | 0.74 | -0.03 points |

**Conclusion**: 384-dim model is 4-5x faster with only 5% quality drop → **best default choice**

## How Embeddings Are Generated

### 1. Tokenization

Text → Token IDs:

```python
text = "Machine learning is fascinating"
# Tokenize
tokens = ["machine", "learning", "is", "fascinating"]
# Map to IDs
token_ids = [4829, 2967, 2003, 10383]
```

### 2. Transformer Encoding

Token IDs → Contextual embeddings:

```python
# Each token gets contextual representation
# Shape: (sequence_length, hidden_size)
token_embeddings = transformer(token_ids)
# Output: (4, 384) for 4 tokens, 384 dims
```

### 3. Pooling

Token embeddings → Sentence embedding:

```python
# Mean pooling (average all token embeddings)
sentence_embedding = torch.mean(token_embeddings, dim=0)
# Output: (384,) single vector for entire sentence
```

### 4. Normalization

```python
# L2 normalization (unit vector)
norm = torch.linalg.norm(sentence_embedding)
sentence_embedding = sentence_embedding / norm
# Now: ||embedding|| = 1.0
```

## Embedding Pipeline in This Repository

### For Document Ingestion

```python
from dartboard.embeddings import SentenceTransformerModel
from dartboard.ingestion.loaders import PDFLoader
from dartboard.datasets.models import Chunk

# 1. Load documents
loader = PDFLoader()
documents = loader.load("document.pdf")

# 2. Chunk documents
from dartboard.ingestion.chunking import SentenceChunker
chunker = SentenceChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(documents[0].content)

# 3. Generate embeddings
model = SentenceTransformerModel("all-MiniLM-L6-v2")
chunk_objects = []

for i, chunk_text in enumerate(chunks):
    embedding = model.encode(chunk_text)  # (384,) numpy array
    chunk = Chunk(
        id=f"doc_{i}",
        text=chunk_text,
        embedding=embedding
    )
    chunk_objects.append(chunk)

# 4. Store in vector database
from dartboard.storage.vector_store import FAISSVectorStore
vector_store = FAISSVectorStore(embedding_dim=384)
vector_store.add(chunk_objects)
```

### For Query Processing

```python
# Query → Embedding → Retrieval
query = "What is machine learning?"
query_embedding = model.encode(query)  # (384,)

# Similarity search
results = vector_store.similarity_search(
    query_embedding=query_embedding,
    k=10
)

# Results ranked by cosine similarity
for chunk in results["chunks"]:
    print(f"Score: {chunk.score:.4f}")
    print(f"Text: {chunk.text}\n")
```

## Alternative Embedding Models

### By Use Case

#### High Quality (Research, Analysis)

```python
# Best accuracy, slower
model = SentenceTransformerModel("all-mpnet-base-v2")
# Dims: 768, Speed: 3K sent/sec, Quality: Excellent
```

#### Multi-Lingual

```python
# 50+ languages
model = SentenceTransformerModel(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
# Dims: 384, Languages: 50+
```

#### Domain-Specific

**Scientific Papers**:
```python
model = SentenceTransformerModel("allenai-specter")
# Trained on citation graphs, paper abstracts
```

**Code Search**:
```python
model = SentenceTransformerModel("code-search-net")
# Trained on GitHub code, docstrings
```

**Question Answering**:
```python
model = SentenceTransformerModel("multi-qa-mpnet-base-dot-v1")
# Optimized for Q&A tasks
```

### By Performance Needs

**Maximum Speed** (Embedded devices, real-time):
```python
model = SentenceTransformerModel("all-MiniLM-L6-v2")
# 384 dims, 14K sent/sec CPU
```

**Balanced** (Recommended default):
```python
model = SentenceTransformerModel("all-MiniLM-L12-v2")
# 384 dims, 8K sent/sec CPU, slightly better quality
```

**Maximum Quality** (Offline processing):
```python
model = SentenceTransformerModel("all-mpnet-base-v2")
# 768 dims, 3K sent/sec CPU, best accuracy
```

## Embedding Properties

### Dimensionality

Higher dimensions → more expressiveness, but:
- Larger storage requirements
- Slower similarity computation
- Risk of curse of dimensionality

| Dimensions | Storage (100K docs) | Similarity Speed | Quality |
|------------|---------------------|------------------|---------|
| 384 | 153 MB | Fast | Good |
| 768 | 307 MB | Medium | Excellent |
| 1024 | 410 MB | Slow | Excellent |

**Default**: 384 dims (good balance)

### Normalization

All embeddings are L2-normalized (unit vectors):

```python
# After encoding
assert np.allclose(np.linalg.norm(embedding), 1.0)
```

**Benefits**:
- Cosine similarity = dot product (faster computation)
- Comparable across different texts
- Prevents magnitude from affecting similarity

### Similarity Metrics

**Cosine Similarity** (used by default):
```python
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
# Range: [-1, 1], higher = more similar
# For normalized vectors: similarity = np.dot(emb1, emb2)
```

**Dot Product** (fast, for normalized vectors):
```python
similarity = np.dot(emb1, emb2)
# Range: [-1, 1], higher = more similar
# Same as cosine for unit vectors
```

**Euclidean Distance** (geometric distance):
```python
distance = np.linalg.norm(emb1 - emb2)
# Range: [0, 2√2], lower = more similar
# Use: distance < threshold
```

## Batch Processing

### Efficient Encoding

```python
# Encode multiple texts at once (much faster)
texts = [chunk1.text, chunk2.text, ..., chunk1000.text]

# Batch encoding
embeddings = model.encode(texts, batch_size=32)
# Shape: (1000, 384)

# Update chunks
for chunk, embedding in zip(chunks, embeddings):
    chunk.embedding = embedding
```

### GPU Acceleration

```python
import torch

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformerModel("all-MiniLM-L6-v2", device=device)

# Encode on GPU (10x faster)
embeddings = model.encode(texts, batch_size=64)
# Speed: ~50K sentences/sec on NVIDIA T4
```

### Progress Tracking

```python
from tqdm import tqdm

# Show progress bar
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True  # tqdm progress bar
)
```

## Embedding Caching

### Why Cache?

- Embedding generation is expensive (1-10ms per doc)
- Documents rarely change
- Queries change frequently → only cache docs

### Implementation

```python
import pickle
from pathlib import Path

# Generate and cache embeddings
embedding_cache_file = Path("embeddings_cache.pkl")

if embedding_cache_file.exists():
    # Load cached embeddings
    with open(embedding_cache_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    print(f"Loaded {len(embeddings_dict)} cached embeddings")
else:
    # Generate embeddings
    embeddings_dict = {}
    for chunk in chunks:
        embeddings_dict[chunk.id] = model.encode(chunk.text)

    # Save cache
    with open(embedding_cache_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

# Use cached embeddings
for chunk in chunks:
    chunk.embedding = embeddings_dict[chunk.id]
```

## Fine-Tuning Embeddings (Advanced)

### When to Fine-Tune

✅ **Domain-specific vocabulary**: Medical, legal, financial terminology
✅ **Poor out-of-the-box performance**: <60% accuracy on your data
✅ **Have training data**: 1000+ query-document pairs with labels

❌ **General-purpose search**: Pre-trained models work well
❌ **Limited training data**: <500 examples won't improve much
❌ **Resource constraints**: Fine-tuning requires GPU, time, expertise

### Fine-Tuning Process

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Prepare training data
train_examples = [
    InputExample(texts=["query", "positive doc"], label=1.0),
    InputExample(texts=["query", "negative doc"], label=0.0),
    # ... 1000+ examples
]

# 2. Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Define training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 4. Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="models/finetuned"
)

# 5. Use fine-tuned model
model = SentenceTransformerModel("models/finetuned")
```

## Troubleshooting

### Issue: Slow Embedding Generation

**Solutions**:
```python
# 1. Use GPU
model = SentenceTransformerModel("all-MiniLM-L6-v2", device="cuda")

# 2. Increase batch size
embeddings = model.encode(texts, batch_size=64)

# 3. Use smaller model
model = SentenceTransformerModel("all-MiniLM-L6-v2")  # vs L12

# 4. Cache embeddings (see Caching section)
```

### Issue: Out of Memory

**Solutions**:
```python
# 1. Process in smaller batches
batch_size = 16  # Reduce from 32/64

# 2. Use smaller model
model = SentenceTransformerModel("all-MiniLM-L6-v2")  # 384 dims

# 3. Free memory after batches
import gc
for i in range(0, len(texts), 1000):
    batch = texts[i:i+1000]
    embeddings = model.encode(batch)
    # ... save embeddings ...
    gc.collect()
```

### Issue: Poor Retrieval Quality

**Solutions**:
```python
# 1. Try larger model
model = SentenceTransformerModel("all-mpnet-base-v2")

# 2. Use domain-specific model
model = SentenceTransformerModel("allenai-specter")  # For papers

# 3. Fine-tune on your data (see Fine-Tuning section)

# 4. Combine with BM25 (hybrid retrieval)
```

## References

### Key Papers

1. **Sentence-BERT**: Reimers & Gurevych (2019)
   - Siamese networks for sentence embeddings

2. **Dense Passage Retrieval**: Karpukhin et al. (2020)
   - Dense embeddings for retrieval

### Model Hub

- **Sentence Transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **HuggingFace Models**: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
- **Model Selection Guide**: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)

### Implementation

- **Source Code**: [dartboard/embeddings.py](../dartboard/embeddings.py)
- **Tests**: [tests/test_embeddings.py](../tests/test_embeddings.py)

## Summary

Embeddings are the foundation of semantic search in RAG systems. The **all-MiniLM-L6-v2** model provides an excellent balance of speed, quality, and resource efficiency, making it the ideal default choice.

**Key Takeaways**:
- ✅ Use **all-MiniLM-L6-v2** as default (384 dims, fast, good quality)
- ✅ **Batch encoding** with GPU for 10x speedup
- ✅ **Cache** document embeddings (only encode once)
- ✅ Use **cosine similarity** for semantic matching
- ✅ **Fine-tune** only if domain-specific and have training data
- ✅ Combine with BM25 in **hybrid mode** for best results
