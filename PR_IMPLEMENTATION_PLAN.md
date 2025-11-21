# Dartboard RAG - Pull Request Implementation Plan

## Overview

Breaking down the RAG system implementation into 8 focused, reviewable PRs that build on each other incrementally.

---

## PR #1: Document Loaders Foundation ✅ READY TO MERGE

**Branch:** `feat/document-loaders`  
**Files Changed:** 3 new files  
**Lines of Code:** ~400 LOC  
**Estimated Review Time:** 30 minutes

### What's Included
- ✅ `dartboard/ingestion/loaders.py` - PDF, Markdown, Code loaders
- ✅ `dartboard/ingestion/__init__.py` - Module init
- ✅ `test_loaders.py` - Comprehensive tests (4/4 passing)

### Dependencies
- pypdf (for PDF parsing)
- pyyaml (for YAML frontmatter)

### Testing
```bash
# Run tests
python test_loaders.py

# Expected output
# 4/4 tests passed
# - Markdown Loader: ✓
# - Code Repository Loader: ✓
# - Directory Loader: ✓
# - PDF Error Handling: ✓
```

### Why Merge First
- Zero dependencies on other PRs
- Fully tested and working
- Enables document ingestion pipeline
- No breaking changes

---

## PR #2: Chunking Pipeline

**Branch:** `feat/chunking-pipeline`  
**Depends On:** PR #1  
**Files Changed:** 2 new files  
**Lines of Code:** ~300 LOC  
**Estimated Review Time:** 45 minutes  
**Effort:** 2 days

### What's Included
- `dartboard/ingestion/chunking.py`
  - `RecursiveChunker` - Sentence-aware chunking with overlap
  - `SemanticChunker` - Paragraph/section-based chunking
  - `FixedSizeChunker` - Simple token-based chunking
  - Token counting utilities
- `tests/test_chunking.py`
  - Test overlap handling
  - Test chunk size limits
  - Test metadata preservation

### Key Features
```python
class RecursiveChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        # Split on sentence boundaries
        # Create overlapping windows
        # Preserve code block integrity
        # Auto-generate embeddings
        pass
```

### Dependencies
- tiktoken (for token counting)
- nltk or spacy (for sentence splitting)

### Testing Strategy
```python
# Test cases
- Single paragraph → multiple chunks with overlap
- Code file → preserve function boundaries
- Markdown → respect heading structure
- Edge cases: very short docs, very long docs
```

### Success Criteria
- ✅ Chunks respect sentence boundaries
- ✅ Overlap preserves context
- ✅ Code blocks not split mid-function
- ✅ Metadata preserved (source, page, section)

---

## PR #3: LLM Integration

**Branch:** `feat/llm-generation`  
**Depends On:** None (independent)  
**Files Changed:** 3 new files  
**Lines of Code:** ~250 LOC  
**Estimated Review Time:** 30 minutes  
**Effort:** 2 days

### What's Included
- `dartboard/generation/generator.py`
  - `RAGGenerator` class
  - OpenAI client wrapper
  - Prompt templates
  - Source citation extraction
- `dartboard/generation/prompts.py`
  - Prompt templates
  - Few-shot examples
- `tests/test_generation.py`
  - Mock LLM responses
  - Test citation extraction
  - Test error handling

### Key Features
```python
class RAGGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(
        self, 
        query: str, 
        chunks: List[Chunk],
        temperature: float = 0.7
    ) -> dict:
        prompt = self._build_prompt(query, chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return self._parse_response(response, chunks)
    
    def _build_prompt(self, query: str, chunks: List[Chunk]) -> str:
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk.text}"
            for i, chunk in enumerate(chunks)
        ])
        return f"Answer using ONLY the context...\n{context}\n\nQ: {query}"
```

### Dependencies
- openai>=1.0.0

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4
```

### Testing Strategy
```python
# Use mock responses to avoid API costs
from unittest.mock import patch

@patch('openai.ChatCompletion.create')
def test_generation(mock_create):
    mock_create.return_value = {
        "choices": [{"message": {"content": "Test answer"}}]
    }
    result = generator.generate(query, chunks)
    assert "answer" in result
    assert "sources" in result
```

### Success Criteria
- ✅ OpenAI integration working
- ✅ Proper prompt formatting
- ✅ Source citations extracted
- ✅ Error handling (API failures, rate limits)

---

## PR #4: End-to-End Ingestion Pipeline

**Branch:** `feat/ingestion-pipeline`  
**Depends On:** PR #1, PR #2  
**Files Changed:** 2 new files  
**Lines of Code:** ~200 LOC  
**Estimated Review Time:** 30 minutes  
**Effort:** 1 day

### What's Included
- `dartboard/ingestion/pipeline.py`
  - `IngestionPipeline` class
  - Orchestrates: Load → Chunk → Embed → Store
  - Batch processing
  - Progress tracking
- `tests/test_pipeline.py`
  - End-to-end ingestion test
  - Test error recovery

### Key Features
```python
class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: ChunkingStrategy,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def ingest(self, source: str) -> dict:
        # Load documents
        docs = self.loader.load(source)
        
        # Chunk
        all_chunks = []
        for doc in docs:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        # Embed
        for chunk in all_chunks:
            if chunk.embedding is None:
                chunk.embedding = self.embedding_model.encode(chunk.text)
        
        # Store
        self.vector_store.add(all_chunks)
        
        return {
            "documents": len(docs),
            "chunks": len(all_chunks),
            "status": "success"
        }
```

### Testing Strategy
```python
def test_end_to_end_ingestion():
    # Create pipeline
    pipeline = IngestionPipeline(
        loader=MarkdownLoader(),
        chunker=RecursiveChunker(chunk_size=512),
        embedding_model=SentenceTransformerModel(),
        vector_store=FAISSStore(embedding_dim=384)
    )
    
    # Ingest test document
    result = pipeline.ingest("test.md")
    
    # Verify
    assert result["documents"] == 1
    assert result["chunks"] > 0
    assert vector_store.count() == result["chunks"]
```

### Success Criteria
- ✅ End-to-end flow working
- ✅ Error handling (failed loads, bad chunks)
- ✅ Batch processing efficient
- ✅ Progress tracking

---

## PR #5: FastAPI Routes (Core Endpoints)

**Branch:** `feat/api-routes-core`  
**Depends On:** PR #2, PR #3, PR #4  
**Files Changed:** 4 new files  
**Lines of Code:** ~400 LOC  
**Estimated Review Time:** 1 hour  
**Effort:** 2 days

### What's Included
- `dartboard/api/routes.py`
  - `/query` endpoint
  - `/ingest` endpoint
  - `/health` endpoint
- `dartboard/api/models.py`
  - Pydantic request/response models
- `dartboard/api/dependencies.py`
  - Dependency injection for services
- `tests/test_api.py`
  - FastAPI test client
  - Test all endpoints

### Key Features
```python
# routes.py
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from dartboard.api.models import QueryRequest, QueryResponse

app = FastAPI(title="Dartboard RAG API")

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator)
):
    # Retrieve chunks
    result = retriever.retrieve(request.query, top_k=request.top_k)
    
    # Generate answer
    answer = generator.generate(request.query, result.chunks)
    
    return QueryResponse(
        answer=answer["answer"],
        sources=answer["sources"],
        retrieval_time_ms=...,
        generation_time_ms=...
    )

@app.post("/ingest")
async def ingest_document(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_pipeline)
):
    # Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Ingest
    result = pipeline.ingest(temp_path)
    
    return {"status": "success", **result}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vector_store_count": ...}
```

### Pydantic Models
```python
# models.py
from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    sigma: float = Field(1.0, gt=0)

class Source(BaseModel):
    text: str
    metadata: dict
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieval_time_ms: float
    generation_time_ms: float
```

### Testing Strategy
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/query", json={
        "query": "What is Dartboard?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
```

### Success Criteria
- ✅ /query endpoint working
- ✅ /ingest endpoint working
- ✅ File upload handling
- ✅ Error responses (400, 404, 500)
- ✅ Request validation

---

## PR #6: Authentication & Rate Limiting

**Branch:** `feat/auth-rate-limiting`  
**Depends On:** PR #5  
**Files Changed:** 3 new files  
**Lines of Code:** ~200 LOC  
**Estimated Review Time:** 30 minutes  
**Effort:** 1 day

### What's Included
- `dartboard/api/auth.py`
  - API key authentication
  - Key validation
  - Key management utilities
- `dartboard/api/middleware.py`
  - Rate limiting middleware
  - Request logging
- `tests/test_auth.py`
  - Test authentication
  - Test rate limiting

### Key Features
```python
# auth.py
from fastapi import Header, HTTPException
from typing import Optional

# Simple in-memory store (replace with DB in production)
VALID_API_KEYS = {
    "sk_test_123": {"name": "Test User", "tier": "free"},
    "sk_prod_456": {"name": "Production User", "tier": "premium"}
}

async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> dict:
    if not x_api_key:
        raise HTTPException(401, "API key required")
    
    key_info = VALID_API_KEYS.get(x_api_key)
    if not key_info:
        raise HTTPException(401, "Invalid API key")
    
    return key_info

# Use as dependency
@app.post("/query")
async def query(
    request: QueryRequest,
    key_info: dict = Depends(verify_api_key)
):
    # key_info contains user metadata
    pass
```

### Rate Limiting
```python
# middleware.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")  # 10 requests per minute
async def query(request: Request, ...):
    pass
```

### Dependencies
- slowapi (for rate limiting)

### Success Criteria
- ✅ API key validation working
- ✅ Rate limiting enforced
- ✅ Clear error messages
- ✅ Different tiers (free vs premium)

---

## PR #7: Monitoring & Observability

**Branch:** `feat/monitoring`  
**Depends On:** PR #5  
**Files Changed:** 2 new files  
**Lines of Code:** ~150 LOC  
**Estimated Review Time:** 20 minutes  
**Effort:** 1 day

### What's Included
- `dartboard/monitoring/metrics.py`
  - Prometheus metrics
  - Custom counters/histograms
- `dartboard/api/routes.py` (updated)
  - `/metrics` endpoint
  - Request timing decorators

### Key Features
```python
# metrics.py
from prometheus_client import Counter, Histogram, generate_latest

# Counters
query_counter = Counter(
    'rag_queries_total',
    'Total queries processed',
    ['status']
)

# Histograms
retrieval_latency = Histogram(
    'retrieval_latency_seconds',
    'Retrieval latency in seconds'
)

generation_latency = Histogram(
    'generation_latency_seconds',
    'Generation latency in seconds'
)

# Decorator for timing
def track_retrieval_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with retrieval_latency.time():
            result = await func(*args, **kwargs)
        return result
    return wrapper

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
```

### Metrics Collected
- `rag_queries_total` - Total queries (by status)
- `retrieval_latency_seconds` - Retrieval time histogram
- `generation_latency_seconds` - Generation time histogram
- `vector_store_size` - Number of chunks in store
- `chunks_retrieved` - Distribution of retrieved chunks

### Dependencies
- prometheus-client

### Success Criteria
- ✅ Prometheus metrics exposed
- ✅ Latency tracking working
- ✅ Metrics endpoint functional
- ✅ Grafana dashboard (optional)

---

## PR #8: Docker Deployment

**Branch:** `feat/docker-deployment`  
**Depends On:** All previous PRs  
**Files Changed:** 4 new files  
**Lines of Code:** ~100 LOC  
**Estimated Review Time:** 30 minutes  
**Effort:** 1 day

### What's Included
- `Dockerfile` - Multi-stage build
- `docker-compose.yml` - Full stack
- `.dockerignore` - Exclude unnecessary files
- `docker/entrypoint.sh` - Startup script
- `README_DEPLOYMENT.md` - Deployment docs

### Dockerfile
```dockerfile
FROM python:3.13-slim as base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY dartboard/ ./dartboard/
COPY *.py ./

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "dartboard.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_STORE_TYPE=faiss
    volumes:
      - ./data:/app/data
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Usage
```bash
# Build
docker-compose build

# Run
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### Success Criteria
- ✅ Docker image builds successfully
- ✅ Container runs without errors
- ✅ API accessible on port 8000
- ✅ Environment variables working
- ✅ Volume mounts preserve data

---

## Summary Timeline

| PR | Title | Effort | Dependencies | Ready To Start |
|----|-------|--------|--------------|----------------|
| #1 | Document Loaders | ✅ Done | None | ✅ NOW |
| #2 | Chunking Pipeline | 2 days | PR #1 | After #1 |
| #3 | LLM Integration | 2 days | None | ✅ NOW |
| #4 | Ingestion Pipeline | 1 day | #1, #2 | After #1, #2 |
| #5 | FastAPI Routes | 2 days | #2, #3, #4 | After #2, #3, #4 |
| #6 | Auth & Rate Limiting | 1 day | #5 | After #5 |
| #7 | Monitoring | 1 day | #5 | After #5 |
| #8 | Docker Deployment | 1 day | All | After all |

**Total Effort:** 10 days (2 weeks)  
**Parallelizable:** PRs #2 and #3 can be done simultaneously

---

## Recommended Merge Order

### Week 1
1. **PR #1** - Document Loaders (ready now)
2. **PR #2** + **PR #3** - Chunking + LLM (parallel)
3. **PR #4** - Ingestion Pipeline

### Week 2
4. **PR #5** - FastAPI Routes
5. **PR #6** + **PR #7** - Auth + Monitoring (parallel)
6. **PR #8** - Docker Deployment

---

## PR Template

```markdown
## Description
Brief description of what this PR adds

## Changes
- Added X
- Updated Y
- Fixed Z

## Testing
- [ ] All tests passing
- [ ] Manual testing completed
- [ ] No breaking changes

## Dependencies
- Requires PR #X to be merged first
- New dependencies: list

## How to Test
```bash
# Steps to test
```

## Checklist
- [ ] Code follows Black formatting
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No secrets in code
```

---

*PR Implementation Plan - 2025-11-20*
*8 PRs → Full Production RAG System*
