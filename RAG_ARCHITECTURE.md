# Dartboard RAG System - Architecture Deep Dive

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

    Documents (PDF, Web, MD)
            │
            ▼
    ┌──────────────┐
    │   Loaders    │  ← Parse files, extract text + metadata
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Chunking    │  ← Split into 512-token chunks with overlap
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Embedding   │  ← Generate 384-dim vectors
    └──────┬───────┘
           │
           ├─────────────────┬──────────────────┐
           │                 │                  │
           ▼                 ▼                  ▼
    ┌────────────┐   ┌─────────────┐   ┌──────────────┐
    │   Vector   │   │  Metadata   │   │   Original   │
    │    DB      │   │  PostgreSQL │   │   S3/Disk    │
    │  (FAISS)   │   │             │   │              │
    └────────────┘   └─────────────┘   └──────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

    User Query: "How does Dartboard improve RAG?"
            │
            ▼
    ┌──────────────┐
    │Query Embed   │  ← Convert query to 384-dim vector
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Stage 1: Vector Search (FAISS)      │
    │  • Fast KNN retrieval                │
    │  • Returns top-100 candidates        │
    │  • ~10-20ms latency                  │
    └──────┬───────────────────────────────┘
           │
           │  Candidates: [chunk₁, chunk₂, ..., chunk₁₀₀]
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Stage 2: Dartboard Refinement       │
    │  • Greedy selection algorithm        │
    │  • Maximizes info gain               │
    │  • Natural diversity                 │
    │  • Returns top-5 diverse chunks      │
    │  • ~50-100ms latency                 │
    └──────┬───────────────────────────────┘
           │
           │  Selected: [chunk_A, chunk_B, chunk_C, chunk_D, chunk_E]
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Context Assembly                    │
    │  • Format chunks into prompt         │
    │  • Add source citations              │
    │  • Inject system instructions        │
    └──────┬───────────────────────────────┘
           │
           │  Prompt: "Context:\n[Source 1]: ...\n..."
           │
           ▼
    ┌──────────────────────────────────────┐
    │  LLM Generation (GPT-4/Claude)       │
    │  • Generate answer                   │
    │  • Cite sources                      │
    │  • ~2-5s latency                     │
    └──────┬───────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Response                            │
    │  {                                   │
    │    "answer": "Dartboard improves...",│
    │    "sources": [...],                 │
    │    "confidence": 0.87                │
    │  }                                   │
    └──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                     PERFORMANCE BREAKDOWN                                │
└─────────────────────────────────────────────────────────────────────────┘

Total Latency (p95): ~2.5-5s

┌─────────────────────┬──────────┬─────────┐
│ Component           │ Latency  │ % Total │
├─────────────────────┼──────────┼─────────┤
│ Query Embedding     │   10ms   │   0.4%  │
│ Vector Search       │   20ms   │   0.8%  │
│ Dartboard Selection │   70ms   │   2.8%  │
│ LLM Generation      │ 2400ms   │  96.0%  │
├─────────────────────┼──────────┼─────────┤
│ TOTAL               │ 2500ms   │  100%   │
└─────────────────────┴──────────┴─────────┘

Optimization: 96% of time is LLM generation
→ Use streaming for better UX
→ Consider caching common queries
```

---

## Component Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│                    Dependency Graph                           │
└──────────────────────────────────────────────────────────────┘

FastAPI Application
    │
    ├─→ HybridRetriever
    │       │
    │       ├─→ VectorStore (FAISS/Pinecone)
    │       │       └─→ numpy, faiss-cpu
    │       │
    │       ├─→ DartboardRetriever
    │       │       ├─→ EmbeddingModel
    │       │       │       └─→ sentence-transformers
    │       │       │               └─→ torch
    │       │       │
    │       │       └─→ DartboardConfig
    │       │               └─→ sigma, top_k, triage_k
    │       │
    │       └─→ ChunkStore
    │               └─→ List[Chunk]
    │
    ├─→ RAGGenerator
    │       ├─→ OpenAI/Anthropic Client
    │       │       └─→ openai / anthropic SDK
    │       │
    │       └─→ PromptTemplate
    │
    ├─→ DocumentIngestion
    │       ├─→ DocumentLoaders
    │       │       ├─→ PDFLoader (pypdf2)
    │       │       ├─→ WebLoader (beautifulsoup4)
    │       │       └─→ MarkdownLoader
    │       │
    │       └─→ ChunkingStrategy
    │               └─→ RecursiveChunker
    │
    └─→ Monitoring
            ├─→ PrometheusMetrics
            └─→ Logging
```

---

## Scaling Strategy

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │ API #1 │  │ API #2 │  │ API #3 │  ← Stateless FastAPI instances
    └────┬───┘  └────┬───┘  └────┬───┘
         │            │            │
         └────────────┼────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    ┌─────────────────────────────────┐
    │    Shared Vector Store          │  ← Centralized Pinecone/Weaviate
    │    (Pinecone)                   │
    └─────────────────────────────────┘
```

**Capacity Planning:**
- 1 API instance handles: ~100 QPS
- For 1000 QPS → 10 instances
- Vector store handles: 10K+ QPS

---

## Caching Strategy

```
Query: "What is machine learning?"
    │
    ├─→ Check Redis Cache (Key: hash(query))
    │       │
    │       ├─→ HIT → Return cached response (5ms)
    │       │
    │       └─→ MISS
    │           │
    │           ▼
    │       Retrieve + Generate (2500ms)
    │           │
    │           ▼
    │       Store in Redis (TTL: 1 hour)
    │           │
    │           ▼
    │       Return response
```

**Cache Hit Rate:**
- Target: 30-40% for common queries
- Savings: 2500ms → 5ms (500x faster)

---

## Multi-Tenancy Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Multi-Tenant Design                         │
└──────────────────────────────────────────────────────────────┘

API Request with tenant_id: "acme_corp"
    │
    ▼
┌───────────────────────────────────────┐
│  Namespace Resolution                 │
│  tenant_id → namespace_id             │
└───────────┬───────────────────────────┘
            │
            ▼
┌───────────────────────────────────────┐
│  Vector Store (Namespaced)            │
│                                       │
│  Namespace: acme_corp                 │
│  ├─ Index: acme_vectors               │
│  └─ Chunks: 10,000                    │
│                                       │
│  Namespace: globex_inc                │
│  ├─ Index: globex_vectors             │
│  └─ Chunks: 5,000                     │
└───────────────────────────────────────┘

PostgreSQL:
    tenants
    ├─ id, name, api_key, plan
    └─ ...
    
    documents
    ├─ id, tenant_id, title, source
    └─ ...
    
    chunks
    ├─ id, document_id, text, vector_id
    └─ ...
```

---

## Deployment Architecture

### Option 1: Docker Compose (Development)

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VECTOR_STORE_TYPE=faiss
      - LLM_PROVIDER=openai
    volumes:
      - ./data:/app/data
  
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: rag_metadata
  
  redis:
    image: redis:7
```

### Option 2: Kubernetes (Production)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dartboard-rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: api
        image: dartboard-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: VECTOR_STORE_TYPE
          value: "pinecone"
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
```

---

## Example API Usage

### 1. Ingest Document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@whitepaper.pdf" \
  -F "metadata={\"source\": \"research\", \"author\": \"Dr. Smith\"}"

# Response:
{
  "document_id": "doc_123",
  "chunks_created": 45,
  "status": "success"
}
```

### 2. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Dartboard improve RAG systems?",
    "top_k": 5,
    "sigma": 1.0
  }'

# Response:
{
  "answer": "Dartboard improves RAG systems by using a probabilistic...",
  "sources": [
    {
      "chunk_id": "chunk_456",
      "text": "The Dartboard algorithm...",
      "metadata": {"page": 3, "source": "research"}
    }
  ],
  "retrieval_time_ms": 85.3,
  "generation_time_ms": 2341.7
}
```

### 3. Stream Response

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain machine learning",
    "top_k": 5
  }'

# Response (Server-Sent Events):
data: {"type": "chunk", "content": "Machine"}
data: {"type": "chunk", "content": " learning"}
data: {"type": "chunk", "content": " is"}
...
data: {"type": "done", "sources": [...]}
```

---

## Migration Path

### Phase 1: Prototype (Weeks 1-2)
```
Local Development:
- FAISS vector store (in-memory)
- SentenceTransformer embeddings (local)
- OpenAI GPT-3.5 (API)
- SQLite metadata store
- No authentication
```

### Phase 2: MVP (Weeks 3-4)
```
Single Server Deployment:
- FAISS vector store (persisted to disk)
- SentenceTransformer embeddings
- OpenAI GPT-4 (API)
- PostgreSQL metadata store
- Basic API key auth
- Docker Compose deployment
```

### Phase 3: Production (Weeks 5-8)
```
Cloud Deployment:
- Pinecone vector store (managed)
- OpenAI embeddings (API)
- OpenAI GPT-4 (API)
- RDS PostgreSQL
- Redis caching
- JWT authentication
- Multi-tenancy
- Kubernetes deployment
- Monitoring (Prometheus/Grafana)
```

---

*Architecture Documentation - 2025-11-20*
