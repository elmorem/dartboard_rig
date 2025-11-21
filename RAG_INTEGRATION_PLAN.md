# Dartboard RAG System - Production Integration Plan

## Overview

This document outlines the architecture and implementation steps for integrating the Dartboard retrieval algorithm into a production-ready RAG (Retrieval-Augmented Generation) system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System Architecture                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│  Chunking &  │────▶│   Vector     │
│   Ingestion  │     │  Embedding   │     │   Storage    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     User     │────▶│  Dartboard   │◀────│    Query     │
│    Query     │     │  Retrieval   │     │  Embedding   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Context    │────▶│     LLM      │
                     │  Assembly    │     │  Generation  │
                     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │   Response   │
                                          │   + Sources  │
                                          └──────────────┘
```

---

## Component Breakdown

### 1. Document Ingestion Pipeline

**Purpose:** Load, process, and prepare documents for retrieval

**Components:**

- **Document Loaders** (Focused Scope)
  - PDF parser (pypdf) ✅ IMPLEMENTED
  - Markdown parser with frontmatter support ✅ IMPLEMENTED
  - Code repository parser (Python, JS, TS, etc.) ✅ IMPLEMENTED
  - Directory loader (auto-detects file types) ✅ IMPLEMENTED

- **Preprocessing**
  - Text cleaning (remove artifacts, normalize whitespace)
  - Language detection
  - Metadata extraction (title, author, date, source)

**Implementation:**
```python
# dartboard/ingestion/loaders.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> List[Dict[str, Any]]:
        """Load documents from source."""
        pass

class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        # Parse PDF, extract text + metadata
        pass

class WebLoader(DocumentLoader):
    def load(self, url: str) -> List[Dict[str, Any]]:
        # Scrape web page, extract content
        pass
```

---

### 2. Chunking Strategy

**Purpose:** Split documents into optimal-sized chunks for retrieval

**Strategies:**
- **Fixed-size chunking** (e.g., 512 tokens with 50 token overlap)
- **Semantic chunking** (split on paragraphs, sections)
- **Recursive chunking** (hierarchical: document → section → paragraph)
- **Sentence-based chunking** (use sentence boundaries)

**Considerations:**
- Chunk size: 256-1024 tokens (balance between context and granularity)
- Overlap: 10-20% to preserve context across boundaries
- Metadata preservation (source, section, page number)

**Implementation:**
```python
# dartboard/ingestion/chunking.py
from typing import List
from dartboard.datasets.models import Chunk

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        pass

class RecursiveChunker(ChunkingStrategy):
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        # Split text into chunks with overlap
        # Generate embeddings for each chunk
        # Return Chunk objects
        pass
```

---

### 3. Vector Storage Layer

**Purpose:** Efficiently store and query embeddings

**Options:**

| Database | Pros | Cons | Best For |
|----------|------|------|----------|
| **FAISS** | Fast, local, free | No persistence out-of-box | Prototyping, <1M vectors |
| **Pinecone** | Managed, scalable, fast | Costs money | Production, >1M vectors |
| **Weaviate** | Open-source, self-hosted | Setup complexity | Medium scale, control |
| **Qdrant** | Modern, Rust-based, fast | Newer, smaller ecosystem | High performance needs |
| **Chroma** | Simple, embeddings-native | Limited scale | Dev/testing |
| **pgvector** | Postgres extension, familiar | Slower than specialized DBs | Existing Postgres infra |

**Recommended:** Start with FAISS for prototyping, migrate to Pinecone/Weaviate for production

**Implementation:**
```python
# dartboard/storage/vector_store.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> List[Chunk]:
        """Return top-k most similar chunks."""
        pass

class FAISSStore(VectorStore):
    def __init__(self, embedding_dim: int):
        import faiss
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        self.chunks = []
    
    def add(self, chunks: List[Chunk]) -> None:
        embeddings = np.array([c.embedding for c in chunks])
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, k: int) -> List[Chunk]:
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        return [self.chunks[i] for i in indices[0]]
```

---

### 4. Dartboard Integration

**Purpose:** Use Dartboard algorithm for final retrieval step

**Two-Stage Retrieval:**
1. **Stage 1 (Triage):** Vector store retrieves top-N candidates (e.g., N=100)
2. **Stage 2 (Dartboard):** Apply Dartboard algorithm to select top-K diverse results

**Why Two-Stage?**
- Vector stores are optimized for fast KNN
- Dartboard provides diversity on smaller candidate set
- Combines speed of vector search with quality of Dartboard

**Implementation:**
```python
# dartboard/api/retriever.py
from dartboard.core import DartboardRetriever, DartboardConfig
from dartboard.storage.vector_store import VectorStore
from dartboard.embeddings import EmbeddingModel

class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        dartboard_config: DartboardConfig,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.dartboard = DartboardRetriever(dartboard_config, embedding_model)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        # Stage 1: Fast vector search
        query_embedding = self.embedding_model.encode(query)
        candidates = self.vector_store.search(
            query_embedding,
            k=self.dartboard.config.triage_k
        )
        
        # Stage 2: Dartboard refinement
        result = self.dartboard.retrieve(query, candidates)
        return result.chunks[:top_k]
```

---

### 5. LLM Integration

**Purpose:** Generate responses using retrieved context

**LLM Options:**
- **OpenAI GPT-4/GPT-3.5** (API, high quality)
- **Anthropic Claude** (API, long context)
- **Local LLMs** (Llama 3, Mistral via Ollama/vLLM)

**Prompt Engineering:**
```python
# dartboard/generation/prompts.py
def build_rag_prompt(query: str, chunks: List[Chunk]) -> str:
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk.text}"
        for i, chunk in enumerate(chunks)
    ])
    
    return f"""You are a helpful assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer (cite sources by number):"""
```

**Implementation:**
```python
# dartboard/generation/generator.py
from typing import List, Dict, Any
from dartboard.datasets.models import Chunk

class RAGGenerator:
    def __init__(self, llm_client, model_name: str = "gpt-4"):
        self.client = llm_client
        self.model = model_name
    
    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        prompt = build_rag_prompt(query, chunks)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "text": c.text,
                    "metadata": c.metadata,
                    "chunk_id": c.id
                }
                for c in chunks
            ]
        }
```

---

### 6. API Layer (FastAPI)

**Purpose:** Expose RAG system via REST API

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ingest` | POST | Upload documents |
| `/query` | POST | Ask questions |
| `/chunks/{id}` | GET | Retrieve chunk by ID |
| `/health` | GET | Health check |
| `/metrics` | GET | System metrics |

**Implementation:**
```python
# dartboard/api/routes.py
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Dartboard RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    sigma: float = 1.0

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    import time
    
    # Retrieve chunks
    start = time.time()
    chunks = hybrid_retriever.retrieve(request.query, request.top_k)
    retrieval_time = (time.time() - start) * 1000
    
    # Generate response
    start = time.time()
    result = rag_generator.generate(request.query, chunks)
    generation_time = (time.time() - start) * 1000
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
    )

@app.post("/ingest")
async def ingest_document(file: UploadFile):
    # Process uploaded document
    # Chunk, embed, store
    pass
```

---

### 7. Monitoring & Observability

**Purpose:** Track system performance and quality

**Metrics to Track:**
- **Latency:** Retrieval time, generation time, end-to-end
- **Throughput:** Queries per second
- **Quality:** User feedback (thumbs up/down), NDCG on test set
- **Diversity:** Average pairwise distance of retrieved chunks
- **Cost:** LLM API costs, infrastructure costs

**Tools:**
- **Prometheus** + **Grafana** for metrics
- **LangSmith** / **LangFuse** for LLM tracing
- **Sentry** for error tracking

**Implementation:**
```python
# dartboard/monitoring/metrics.py
from prometheus_client import Counter, Histogram

query_counter = Counter('rag_queries_total', 'Total queries')
retrieval_latency = Histogram('retrieval_latency_seconds', 'Retrieval time')
generation_latency = Histogram('generation_latency_seconds', 'Generation time')

@retrieval_latency.time()
def timed_retrieve(query: str) -> List[Chunk]:
    query_counter.inc()
    return retriever.retrieve(query)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Set up FastAPI application structure
- [ ] Implement document loaders (PDF, Markdown, Web)
- [ ] Create chunking pipeline
- [ ] Integrate FAISS vector store
- [ ] Wire up Dartboard retriever
- [ ] Basic `/query` and `/ingest` endpoints

### Phase 2: LLM Integration (Week 3)
- [ ] Integrate OpenAI/Claude API
- [ ] Implement prompt engineering
- [ ] Add source citation
- [ ] Response streaming support

### Phase 3: Production Readiness (Week 4)
- [ ] Add authentication (API keys)
- [ ] Implement rate limiting
- [ ] Add caching layer (Redis)
- [ ] Set up monitoring (Prometheus)
- [ ] Write deployment docs (Docker, K8s)

### Phase 4: Advanced Features (Week 5+)
- [ ] Hybrid search (BM25 + vector)
- [ ] Query rewriting/expansion
- [ ] Conversation history support
- [ ] Multi-tenant support
- [ ] A/B testing framework

---

## Technology Stack Recommendation

```yaml
Backend:
  Framework: FastAPI 0.116+
  Runtime: Python 3.13 + uv
  
Retrieval:
  Embedding Model: all-MiniLM-L6-v2 (or OpenAI text-embedding-3-large)
  Vector Store: FAISS (dev) → Pinecone (prod)
  Dartboard: Custom implementation (already built)

Generation:
  LLM: OpenAI GPT-4 or Anthropic Claude 3.5
  
Storage:
  Metadata DB: PostgreSQL 16
  Cache: Redis 7
  Object Storage: S3 (for documents)

Monitoring:
  Metrics: Prometheus + Grafana
  Tracing: LangSmith or LangFuse
  Logs: CloudWatch / ELK Stack

Deployment:
  Containerization: Docker
  Orchestration: Kubernetes or ECS
  CI/CD: GitHub Actions
```

---

## File Structure

```
vastai/
├── dartboard/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # FastAPI endpoints
│   │   └── dependencies.py    # DI for retriever, etc.
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py         # Document loaders
│   │   └── chunking.py        # Chunking strategies
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # Vector DB abstraction
│   │   └── metadata_store.py  # PostgreSQL for metadata
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py         # Prompt templates
│   │   └── generator.py       # LLM wrapper
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── metrics.py         # Prometheus metrics
│   ├── core.py                # Dartboard algorithm (existing)
│   ├── embeddings.py          # (existing)
│   └── utils.py               # (existing)
├── config/
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
├── tests/
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Configuration Management

**Environment-based config:**
```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Dartboard
    dartboard_sigma: float = 1.0
    dartboard_top_k: int = 5
    dartboard_triage_k: int = 100
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Vector Store
    vector_store_type: str = "faiss"  # faiss, pinecone, weaviate
    pinecone_api_key: str = ""
    
    # LLM
    llm_provider: str = "openai"  # openai, anthropic
    openai_api_key: str = ""
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Cost Estimation (Monthly)

**Assumptions:** 10K queries/day, 100K documents

| Component | Service | Cost |
|-----------|---------|------|
| Vector Store | Pinecone (1 pod) | $70 |
| LLM API | OpenAI GPT-4 (10K queries) | $300 |
| Embeddings | OpenAI embeddings | $20 |
| Compute | AWS EC2 (t3.large) | $60 |
| Database | RDS PostgreSQL | $40 |
| **Total** | | **~$490/month** |

**Optimizations:**
- Use GPT-3.5 instead of GPT-4: Save $200/month
- Self-hosted embeddings (SentenceTransformer): Save $20/month
- FAISS instead of Pinecone: Save $70/month (if <100K vectors)

---

## Security Considerations

1. **API Authentication**
   - JWT tokens or API keys
   - Rate limiting per user/key

2. **Data Privacy**
   - Encrypt documents at rest (S3 encryption)
   - Encrypt embeddings in vector store
   - PII detection/masking

3. **Access Control**
   - Role-based access (admin, user, readonly)
   - Document-level permissions

4. **Input Validation**
   - Sanitize user queries
   - File upload validation (type, size)
   - Prompt injection protection

---

## Testing Strategy

1. **Unit Tests**
   - Test chunking logic
   - Test Dartboard retrieval
   - Test prompt generation

2. **Integration Tests**
   - Test full pipeline (ingest → retrieve → generate)
   - Test API endpoints

3. **Evaluation**
   - NDCG, MAP on benchmark datasets
   - Diversity metrics
   - Human evaluation (quality scoring)

4. **Load Testing**
   - Locust or k6 for API load testing
   - Target: 100 QPS with <500ms p95 latency

---

## Next Steps

1. **Decide on LLM provider** (OpenAI vs Anthropic vs local)
2. **Choose vector store** (FAISS for MVP vs Pinecone for production)
3. **Define data sources** (what documents to ingest?)
4. **Set up development environment** (Docker Compose)
5. **Implement Phase 1** (core infrastructure)

---

## Questions to Answer

- [ ] What types of documents will be ingested?
- [ ] What is the expected query volume?
- [ ] Do we need conversation history/multi-turn support?
- [ ] What level of customization do users need (σ, top_k)?
- [ ] Self-hosted vs managed services preference?
- [ ] Budget constraints?
- [ ] Compliance requirements (GDPR, HIPAA)?

---

*Plan created: 2025-11-20*
*Dartboard RAG System Integration*
