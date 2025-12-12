# Dartboard RAG API Guide

## Overview

The Dartboard RAG API is a FastAPI-based REST API for Retrieval-Augmented Generation. It provides endpoints for querying your knowledge base, ingesting documents, comparing retrieval methods, and monitoring system health.

**Base URL:** `http://localhost:8000` (default)

**Key Features:**
- 🔍 **Query Endpoint**: RAG-based question answering with source citations
- 📤 **Ingest Endpoint**: Upload and index documents (PDF, Markdown, text)
- 🔬 **Compare Endpoint**: Benchmark different retrieval methods side-by-side
- ❤️ **Health Endpoint**: System status and configuration
- 📊 **Metrics Endpoint**: Prometheus monitoring metrics
- 🔐 **API Key Authentication**: Tiered access control
- ⚡ **Rate Limiting**: Automatic request throttling
- 📝 **Request Logging**: Comprehensive request/response logging

## Quick Start

### Starting the API

```bash
# Install dependencies
pip install fastapi uvicorn

# Set environment variables
export OPENAI_API_KEY="sk-..."
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Run server
uvicorn dartboard.api.routes:app --reload

# Server starts at http://localhost:8000
```

### First API Call

```bash
# Query the API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_test_123456" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5
  }'
```

### Interactive Documentation

FastAPI provides automatic interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

### API Keys

All protected endpoints require an API key in the `X-API-Key` header.

**Request:**
```http
POST /query HTTP/1.1
Host: localhost:8000
Content-Type: application/json
X-API-Key: sk_test_123456
```

### Default API Keys

For development/testing:

| API Key | Name | Tier | Rate Limit |
|---------|------|------|------------|
| `sk_test_123456` | Test User | free | 10 req/min |
| `sk_prod_789012` | Production User | premium | 100 req/min |

### API Key Tiers

| Tier | Rate Limit | Use Case |
|------|------------|----------|
| **free** | 10 req/min | Development, testing, small projects |
| **premium** | 100 req/min | Production applications, higher volume |
| **enterprise** | 1000 req/min | Large-scale deployments |

### Error Responses

**Missing API Key:**
```json
{
  "detail": "API key required. Include X-API-Key header in your request."
}
```
Status: `401 Unauthorized`

**Invalid API Key:**
```json
{
  "detail": "Invalid API key. Check your credentials."
}
```
Status: `401 Unauthorized`

**Inactive API Key:**
```json
{
  "detail": "API key has been deactivated. Contact support."
}
```
Status: `401 Unauthorized`

## Endpoints

### POST /query

Answer questions using RAG with source citations.

**Request:**
```json
{
  "query": "What is the attention mechanism?",
  "top_k": 5,
  "sigma": 1.0,
  "temperature": 0.7
}
```

**Parameters:**
- `query` (string, required): User question
- `top_k` (int, optional): Number of chunks to retrieve (1-20, default: 5)
- `sigma` (float, optional): Dartboard temperature parameter (>0, default: 1.0)
- `temperature` (float, optional): LLM generation temperature (0-2, default: 0.7)

**Response:**
```json
{
  "answer": "The attention mechanism allows models to focus on relevant parts...",
  "sources": [
    {
      "number": 1,
      "text": "Attention mechanism allows neural networks...",
      "metadata": {
        "source": "paper.pdf",
        "page": 3
      },
      "chunk_id": "chunk_42"
    }
  ],
  "num_sources_cited": 2,
  "retrieval_time_ms": 45.23,
  "generation_time_ms": 1203.45,
  "total_time_ms": 1248.68,
  "model": "gpt-3.5-turbo"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_test_123456" \
  -d '{
    "query": "What is deep learning?",
    "top_k": 5,
    "temperature": 0.5
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    headers={"X-API-Key": "sk_test_123456"},
    json={
        "query": "What is deep learning?",
        "top_k": 5,
        "temperature": 0.5
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources_cited']}")
print(f"Time: {result['total_time_ms']}ms")
```

**JavaScript Example:**
```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'sk_test_123456'
  },
  body: JSON.stringify({
    query: 'What is deep learning?',
    top_k: 5,
    temperature: 0.5
  })
});

const result = await response.json();
console.log('Answer:', result.answer);
console.log('Sources:', result.num_sources_cited);
```

### POST /ingest

Upload and ingest documents into the vector store.

**Request:**
```http
POST /ingest HTTP/1.1
Content-Type: multipart/form-data
X-API-Key: sk_test_123456

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

<binary data>
------WebKitFormBoundary
Content-Disposition: form-data; name="chunk_size"

512
------WebKitFormBoundary
Content-Disposition: form-data; name="overlap"

50
------WebKitFormBoundary--
```

**Form Parameters:**
- `file` (file, required): Document to ingest (.pdf, .md, .txt)
- `chunk_size` (int, optional): Max tokens per chunk (100-2000, default: 512)
- `overlap` (int, optional): Token overlap between chunks (0-500, default: 50)
- `track_progress` (bool, optional): Enable progress logging (default: false)

**Response:**
```json
{
  "status": "success",
  "documents_processed": 1,
  "chunks_created": 42,
  "chunks_stored": 42,
  "processing_time_ms": 3456.78,
  "errors": [],
  "metadata": {
    "source": "/tmp/document.pdf",
    "avg_chunk_size": 387.5,
    "filename": "document.pdf",
    "file_size": 125630
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "X-API-Key: sk_test_123456" \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "overlap=50"
```

**Python Example:**
```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ingest",
        headers={"X-API-Key": "sk_test_123456"},
        files={"file": f},
        data={
            "chunk_size": 512,
            "overlap": 50,
            "track_progress": True
        }
    )

result = response.json()
print(f"Status: {result['status']}")
print(f"Chunks created: {result['chunks_created']}")
print(f"Processing time: {result['processing_time_ms']}ms")
```

**Supported File Types:**
- `.pdf` - PDF documents
- `.md` - Markdown files
- `.txt` - Plain text files

**Error Responses:**

**Unsupported file type:**
```json
{
  "detail": "Unsupported file type: .docx. Allowed: .md, .txt, .pdf"
}
```
Status: `400 Bad Request`

### POST /compare

Compare multiple retrieval methods side-by-side.

**Request:**
```json
{
  "query": "What is attention mechanism?",
  "methods": ["bm25", "dense", "hybrid", "dartboard"],
  "top_k": 5,
  "use_reranker": false
}
```

**Parameters:**
- `query` (string, required): Search query
- `methods` (array, optional): Retrieval methods to compare (default: ["bm25", "dense", "hybrid"])
  - `bm25`: BM25 sparse retrieval
  - `dense`: Dense vector similarity
  - `hybrid`: BM25 + Dense with RRF fusion
  - `dartboard`: Dartboard DPP algorithm
- `top_k` (int, optional): Number of results per method (1-20, default: 5)
- `use_reranker` (bool, optional): Apply cross-encoder reranking (default: false)

**Response:**
```json
{
  "query": "What is attention mechanism?",
  "results": [
    {
      "method": "bm25",
      "chunks": [
        {
          "id": "chunk_1",
          "text": "...",
          "metadata": {},
          "score": 0.85
        }
      ],
      "scores": [0.85, 0.72, 0.68, 0.55, 0.42],
      "latency_ms": 12.34,
      "metadata": {}
    },
    {
      "method": "dense",
      "chunks": [...],
      "scores": [0.92, 0.88, 0.81, 0.75, 0.70],
      "latency_ms": 45.67,
      "metadata": {}
    },
    {
      "method": "hybrid",
      "chunks": [...],
      "scores": [0.90, 0.85, 0.82, 0.79, 0.74],
      "latency_ms": 58.90,
      "metadata": {"rrf_k": 60}
    }
  ],
  "total_time_ms": 123.45,
  "reranker_used": false,
  "comparison_metrics": {
    "total_unique_chunks": 12,
    "methods_compared": 3,
    "avg_latency_ms": 38.97,
    "overlap": {
      "bm25_vs_dense": {
        "overlap_count": 2,
        "overlap_percentage": 40.0
      },
      "bm25_vs_hybrid": {
        "overlap_count": 3,
        "overlap_percentage": 60.0
      },
      "dense_vs_hybrid": {
        "overlap_count": 4,
        "overlap_percentage": 80.0
      }
    }
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_test_123456" \
  -d '{
    "query": "What is attention?",
    "methods": ["bm25", "dense", "hybrid"],
    "top_k": 5,
    "use_reranker": true
  }'
```

**Python Example:**
```python
response = requests.post(
    "http://localhost:8000/compare",
    headers={"X-API-Key": "sk_test_123456"},
    json={
        "query": "What is attention?",
        "methods": ["bm25", "dense", "hybrid", "dartboard"],
        "top_k": 5,
        "use_reranker": True
    }
)

result = response.json()
for method_result in result["results"]:
    print(f"{method_result['method']}: {method_result['latency_ms']}ms")
    print(f"  Scores: {method_result['scores']}")
```

### GET /health

System health check and configuration.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_count": 1250,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "gpt-3.5-turbo",
  "version": "1.0.0"
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/health"
```

**No authentication required.**

### GET /metrics

Prometheus metrics for monitoring.

**Response** (Prometheus text format):
```
# HELP rag_queries_total Total number of RAG queries processed
# TYPE rag_queries_total counter
rag_queries_total{status="success",tier="free"} 42.0

# HELP rag_retrieval_latency_seconds Time spent retrieving chunks
# TYPE rag_retrieval_latency_seconds histogram
rag_retrieval_latency_seconds_bucket{le="0.01"} 5.0
rag_retrieval_latency_seconds_bucket{le="0.05"} 32.0
...

# HELP rag_vector_store_size Current number of chunks in vector store
# TYPE rag_vector_store_size gauge
rag_vector_store_size 1250.0
```

**cURL Example:**
```bash
curl "http://localhost:8000/metrics"
```

**No authentication required.**

See [Monitoring](./monitoring.md) for details on available metrics.

### GET /

Root endpoint with API information.

**Response:**
```json
{
  "name": "Dartboard RAG API",
  "version": "1.0.0",
  "description": "Retrieval-Augmented Generation with Dartboard algorithm",
  "endpoints": {
    "query": "POST /query - Answer questions using RAG",
    "compare": "POST /compare - Compare multiple retrieval methods",
    "ingest": "POST /ingest - Ingest documents",
    "health": "GET /health - Health check",
    "metrics": "GET /metrics - Prometheus metrics",
    "docs": "GET /docs - Interactive API documentation"
  }
}
```

**No authentication required.**

## Rate Limiting

All requests are rate-limited based on API key tier or IP address.

### Rate Limit Headers

Every response includes rate limit information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1672531260
X-Process-Time: 1248.68ms
```

**Headers:**
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets
- `X-Process-Time`: Request processing time

### Rate Limit Exceeded

When rate limit is exceeded:

**Response:**
```json
{
  "detail": {
    "error": "Rate limit exceeded",
    "message": "Too many requests. Limit: 10/minute",
    "retry_after": 45,
    "rate_limit": {
      "limit": 10,
      "remaining": 0,
      "reset": 1672531260,
      "used": 10
    }
  }
}
```

**Status:** `429 Too Many Requests`

**Headers:**
```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1672531260
Retry-After: 45
```

### Exempt Endpoints

These endpoints are **not** rate-limited:
- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /docs`
- `GET /openapi.json`
- `GET /redoc`

## Error Handling

### Standard Error Response

```json
{
  "error": "Error Type",
  "detail": "Detailed error message",
  "status_code": 400
}
```

### Common Error Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `400` | Bad Request | Invalid parameters, unsupported file type |
| `401` | Unauthorized | Missing or invalid API key |
| `404` | Not Found | No relevant documents found |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error, check logs |

### Example Error Responses

**No relevant documents:**
```json
{
  "detail": "No relevant documents found in vector store"
}
```
Status: `404 Not Found`

**Invalid parameters:**
```json
{
  "detail": [
    {
      "loc": ["body", "top_k"],
      "msg": "ensure this value is less than or equal to 20",
      "type": "value_error.number.not_le"
    }
  ]
}
```
Status: `422 Unprocessable Entity`

## Client Examples

### Python Client

```python
import requests
from typing import Optional

class DartboardClient:
    """Python client for Dartboard RAG API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })

    def query(
        self,
        query: str,
        top_k: int = 5,
        sigma: float = 1.0,
        temperature: float = 0.7
    ) -> dict:
        """Query the RAG system."""
        response = self.session.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "top_k": top_k,
                "sigma": sigma,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()

    def ingest(
        self,
        file_path: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> dict:
        """Ingest a document."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'chunk_size': chunk_size,
                'overlap': overlap
            }
            response = self.session.post(
                f"{self.base_url}/ingest",
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()

    def compare(
        self,
        query: str,
        methods: list = None,
        top_k: int = 5,
        use_reranker: bool = False
    ) -> dict:
        """Compare retrieval methods."""
        response = self.session.post(
            f"{self.base_url}/compare",
            json={
                "query": query,
                "methods": methods or ["bm25", "dense", "hybrid"],
                "top_k": top_k,
                "use_reranker": use_reranker
            }
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# Usage
client = DartboardClient(
    base_url="http://localhost:8000",
    api_key="sk_test_123456"
)

# Query
result = client.query("What is machine learning?")
print(result["answer"])

# Ingest
result = client.ingest("document.pdf", chunk_size=512)
print(f"Ingested {result['chunks_created']} chunks")

# Compare
result = client.compare(
    "What is attention?",
    methods=["bm25", "dense", "hybrid"]
)
for method in result["results"]:
    print(f"{method['method']}: {method['latency_ms']}ms")
```

### JavaScript/TypeScript Client

```typescript
interface QueryRequest {
  query: string;
  top_k?: number;
  sigma?: number;
  temperature?: number;
}

interface QueryResponse {
  answer: string;
  sources: Array<{
    number: number;
    text: string;
    metadata: Record<string, any>;
  }>;
  num_sources_cited: number;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  model: string;
}

class DartboardClient {
  constructor(
    private baseUrl: string,
    private apiKey: string
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }

    return response.json();
  }

  async query(params: QueryRequest): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async ingest(file: File, options?: {
    chunk_size?: number;
    overlap?: number;
  }): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.chunk_size) {
      formData.append('chunk_size', String(options.chunk_size));
    }
    if (options?.overlap) {
      formData.append('overlap', String(options.overlap));
    }

    const response = await fetch(`${this.baseUrl}/ingest`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Ingestion failed');
    }

    return response.json();
  }

  async health(): Promise<any> {
    return this.request('/health');
  }
}

// Usage
const client = new DartboardClient(
  'http://localhost:8000',
  'sk_test_123456'
);

// Query
const result = await client.query({
  query: 'What is machine learning?',
  top_k: 5,
  temperature: 0.7
});

console.log(result.answer);
console.log(`Sources cited: ${result.num_sources_cited}`);
```

## Deployment

### Production Configuration

```python
# main.py
import uvicorn
from dartboard.api.routes import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for production
        log_level="info",
        access_log=True
    )
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_DIM="384"
export EMBEDDING_DEVICE="cpu"  # or "cuda"

# API Configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_WORKERS="4"
```

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "dartboard.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```bash
# Build and run
docker build -t dartboard-api .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  dartboard-api
```

### CORS Configuration

For production, configure CORS properly:

```python
# dartboard/api/routes.py

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],  # Specific domains instead of ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### HTTPS/TLS

Use a reverse proxy (nginx, Caddy) for HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Best Practices

### 1. Use Persistent Connections

```python
# ✅ Good: Reuse session
session = requests.Session()
session.headers.update({'X-API-Key': api_key})

for query in queries:
    response = session.post(url, json={"query": query})

# ❌ Bad: New connection each time
for query in queries:
    response = requests.post(
        url,
        headers={'X-API-Key': api_key},
        json={"query": query}
    )
```

### 2. Handle Rate Limits

```python
import time

def query_with_retry(client, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.query(query)
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### 3. Batch Operations

```python
# Ingest multiple documents
for doc in documents:
    try:
        result = client.ingest(doc)
        print(f"✓ {doc}: {result['chunks_created']} chunks")
    except Exception as e:
        print(f"✗ {doc}: {e}")
        continue
```

### 4. Monitor Response Times

```python
import time

start = time.time()
result = client.query(query)
elapsed = time.time() - start

print(f"Total time: {elapsed*1000:.2f}ms")
print(f"Server time: {result['total_time_ms']}ms")
print(f"Network overhead: {(elapsed*1000 - result['total_time_ms']):.2f}ms")
```

### 5. Validate Responses

```python
def validate_query_response(response):
    assert 'answer' in response
    assert 'sources' in response
    assert response['num_sources_cited'] <= len(response['sources'])
    return response
```

## Troubleshooting

### Issue: Connection refused

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start server
uvicorn dartboard.api.routes:app --reload
```

### Issue: 401 Unauthorized

**Solution:**
```python
# Verify API key is included
headers = {'X-API-Key': 'sk_test_123456'}

# Check API key is valid
response = requests.get("http://localhost:8000/health")
```

### Issue: 404 No relevant documents

**Solution:**
```python
# Ingest documents first
client.ingest("document.pdf")

# Verify vector store has content
health = client.health()
print(f"Vector store size: {health['vector_store_count']}")
```

### Issue: Slow responses

**Causes:**
1. Large `top_k` value
2. Slow LLM API
3. Large documents

**Solutions:**
```python
# Reduce top_k
result = client.query(query, top_k=3)  # Instead of 10

# Use faster model
# Set OPENAI_MODEL="gpt-3.5-turbo" (faster than gpt-4)

# Optimize chunk size
client.ingest(doc, chunk_size=256)  # Smaller chunks = faster retrieval
```

## See Also

- [Retrieval Methods](./retrieval-methods.md) - Understanding retrieval approaches
- [Generation](./generation.md) - LLM generation details
- [Monitoring](./monitoring.md) - Prometheus metrics and dashboards
- [Ingestion Pipeline](./ingestion-pipeline.md) - Document ingestion details
- [Vector Stores](./vector-stores.md) - Storage layer configuration
