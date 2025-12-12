# Monitoring and Observability

## Overview

Dartboard provides comprehensive monitoring through Prometheus metrics, enabling you to track system performance, detect issues, and optimize your RAG application.

**Key Metrics:**
- 📊 **Query Metrics**: Request counts, success/failure rates, error types
- ⏱️ **Latency Tracking**: Retrieval, generation, and total query time
- 📤 **Ingestion Monitoring**: Document processing statistics
- 💾 **Vector Store Health**: Size and retrieval patterns
- 🔐 **Authentication**: Auth attempts and rate limiting
- 📈 **System Info**: Configuration and version tracking

**Metrics Endpoint:** `GET /metrics` (Prometheus text format)

## Quick Start

### Accessing Metrics

```bash
# View metrics in your browser
curl http://localhost:8000/metrics

# Or visit directly
open http://localhost:8000/metrics
```

**Output** (Prometheus text format):
```
# HELP rag_queries_total Total number of RAG queries processed
# TYPE rag_queries_total counter
rag_queries_total{status="success",tier="free"} 42.0
rag_queries_total{status="error",tier="free"} 2.0

# HELP rag_retrieval_latency_seconds Time spent retrieving chunks
# TYPE rag_retrieval_latency_seconds histogram
rag_retrieval_latency_seconds_bucket{le="0.01"} 5.0
rag_retrieval_latency_seconds_bucket{le="0.05"} 32.0
...
rag_retrieval_latency_seconds_sum 12.45
rag_retrieval_latency_seconds_count 42

# HELP rag_vector_store_size Current number of chunks
# TYPE rag_vector_store_size gauge
rag_vector_store_size 1250.0
```

### Prometheus Setup

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'dartboard-rag'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

**Start Prometheus:**
```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

**Access Prometheus UI:** http://localhost:9090

## Available Metrics

### Query Metrics

#### rag_queries_total (Counter)

Total number of RAG queries processed.

**Labels:**
- `status`: "success" or "error"
- `tier`: API key tier ("free", "premium", "enterprise")

**Example:**
```promql
# Total successful queries
rag_queries_total{status="success"}

# Error rate
rate(rag_queries_total{status="error"}[5m])

# Queries per tier
sum by (tier) (rag_queries_total)
```

#### rag_query_errors_total (Counter)

Total query errors by type.

**Labels:**
- `error_type`: Exception class name

**Example:**
```promql
# Most common errors
topk(5, sum by (error_type) (rag_query_errors_total))

# HTTPException rate
rate(rag_query_errors_total{error_type="HTTPException"}[5m])
```

### Latency Metrics

#### rag_retrieval_latency_seconds (Histogram)

Time spent retrieving chunks from vector store.

**Buckets:** 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0 seconds

**Example:**
```promql
# Average retrieval latency
rate(rag_retrieval_latency_seconds_sum[5m]) /
rate(rag_retrieval_latency_seconds_count[5m])

# 95th percentile retrieval time
histogram_quantile(0.95, rate(rag_retrieval_latency_seconds_bucket[5m]))

# Percentage of queries under 100ms
sum(rate(rag_retrieval_latency_seconds_bucket{le="0.1"}[5m])) /
sum(rate(rag_retrieval_latency_seconds_count[5m])) * 100
```

#### rag_generation_latency_seconds (Histogram)

Time spent generating answers with LLM.

**Buckets:** 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0 seconds

**Example:**
```promql
# Average generation time
rate(rag_generation_latency_seconds_sum[5m]) /
rate(rag_generation_latency_seconds_count[5m])

# 99th percentile (slowest queries)
histogram_quantile(0.99, rate(rag_generation_latency_seconds_bucket[5m]))
```

#### rag_total_query_latency_seconds (Histogram)

Total end-to-end query processing time (retrieval + generation).

**Buckets:** 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0 seconds

**Example:**
```promql
# Average total query time
rate(rag_total_query_latency_seconds_sum[5m]) /
rate(rag_total_query_latency_seconds_count[5m])

# SLA: % of queries under 5 seconds
sum(rate(rag_total_query_latency_seconds_bucket{le="5"}[5m])) /
sum(rate(rag_total_query_latency_seconds_count[5m])) * 100
```

### Ingestion Metrics

#### rag_ingestions_total (Counter)

Total document ingestions.

**Labels:**
- `status`: "success" or "error"
- `file_type`: "pdf", "md", "txt"

**Example:**
```promql
# Ingestions per file type
sum by (file_type) (rag_ingestions_total)

# PDF ingestion failure rate
rate(rag_ingestions_total{status="error",file_type="pdf"}[1h])
```

#### rag_ingestion_latency_seconds (Histogram)

Time spent ingesting documents.

**Buckets:** 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0 seconds

**Example:**
```promql
# Average ingestion time
rate(rag_ingestion_latency_seconds_sum[1h]) /
rate(rag_ingestion_latency_seconds_count[1h])
```

#### rag_chunks_created_total (Counter)

Total chunks created from documents.

**Example:**
```promql
# Chunks created per hour
rate(rag_chunks_created_total[1h]) * 3600

# Total chunks created
rag_chunks_created_total
```

#### rag_chunks_stored_total (Counter)

Total chunks stored in vector store.

**Example:**
```promql
# Storage rate
rate(rag_chunks_stored_total[1h]) * 3600
```

### Vector Store Metrics

#### rag_vector_store_size (Gauge)

Current number of chunks in vector store.

**Example:**
```promql
# Current size
rag_vector_store_size

# Growth rate
rate(rag_vector_store_size[1h])

# Alert if empty
rag_vector_store_size == 0
```

#### rag_chunks_retrieved (Histogram)

Number of chunks retrieved per query.

**Buckets:** 1, 3, 5, 10, 15, 20, 30, 50

**Example:**
```promql
# Average chunks per query
rate(rag_chunks_retrieved_sum[5m]) /
rate(rag_chunks_retrieved_count[5m])

# Queries retrieving 10+ chunks
sum(rate(rag_chunks_retrieved_bucket{le="10"}[5m]))
```

### Authentication Metrics

#### rag_auth_attempts_total (Counter)

Total authentication attempts.

**Labels:**
- `status`: "success" or "failure"
- `tier`: API key tier

**Example:**
```promql
# Failed auth rate
rate(rag_auth_attempts_total{status="failure"}[5m])

# Auth success rate
sum(rate(rag_auth_attempts_total{status="success"}[5m])) /
sum(rate(rag_auth_attempts_total[5m])) * 100
```

### Rate Limiting Metrics

#### rag_rate_limit_hits_total (Counter)

Total rate limit violations.

**Labels:**
- `tier`: API key tier

**Example:**
```promql
# Rate limit hits per tier
sum by (tier) (rag_rate_limit_hits_total)

# Free tier rate limit rate
rate(rag_rate_limit_hits_total{tier="free"}[5m])
```

#### rag_requests_by_tier_total (Counter)

Total requests by API key tier.

**Labels:**
- `tier`: API key tier

**Example:**
```promql
# Request distribution
sum by (tier) (rag_requests_by_tier_total)

# Premium tier request rate
rate(rag_requests_by_tier_total{tier="premium"}[5m])
```

### System Information

#### rag_system_info (Info)

System configuration metadata.

**Labels:**
- `version`: API version
- `embedding_model`: Embedding model name
- `llm_model`: LLM model name
- `vector_store_type`: Vector store type

**Example:**
```promql
# System configuration
rag_system_info

# Check version
rag_system_info{version="1.0.0"}
```

## Grafana Dashboards

### Setting Up Grafana

```bash
# Run Grafana with Docker
docker run -d -p 3000:3000 grafana/grafana

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

**Add Prometheus Data Source:**
1. Go to Configuration → Data Sources
2. Add Prometheus
3. URL: `http://localhost:9090`
4. Save & Test

### Dashboard Panels

#### Query Performance Dashboard

**Panel 1: Query Rate**
```promql
sum(rate(rag_queries_total[5m])) by (status)
```
Type: Graph
Title: "Queries per Second"

**Panel 2: Average Query Time**
```promql
rate(rag_total_query_latency_seconds_sum[5m]) /
rate(rag_total_query_latency_seconds_count[5m])
```
Type: Stat
Title: "Avg Query Time"
Unit: seconds

**Panel 3: Latency Breakdown**
```promql
# Retrieval
rate(rag_retrieval_latency_seconds_sum[5m]) /
rate(rag_retrieval_latency_seconds_count[5m])

# Generation
rate(rag_generation_latency_seconds_sum[5m]) /
rate(rag_generation_latency_seconds_count[5m])
```
Type: Bar Chart
Title: "Latency Breakdown"

**Panel 4: Error Rate**
```promql
sum(rate(rag_queries_total{status="error"}[5m])) /
sum(rate(rag_queries_total[5m])) * 100
```
Type: Stat
Title: "Error Rate %"
Thresholds: Green <5%, Yellow 5-10%, Red >10%

#### Vector Store Dashboard

**Panel 1: Vector Store Size**
```promql
rag_vector_store_size
```
Type: Stat
Title: "Total Chunks"

**Panel 2: Ingestion Rate**
```promql
rate(rag_chunks_created_total[1h]) * 3600
```
Type: Graph
Title: "Chunks Created per Hour"

**Panel 3: Average Chunks Retrieved**
```promql
rate(rag_chunks_retrieved_sum[5m]) /
rate(rag_chunks_retrieved_count[5m])
```
Type: Stat
Title: "Avg Chunks per Query"

#### Resource Usage Dashboard

**Panel 1: Request Distribution by Tier**
```promql
sum by (tier) (rate(rag_requests_by_tier_total[5m]))
```
Type: Pie Chart
Title: "Requests by Tier"

**Panel 2: Rate Limit Violations**
```promql
sum by (tier) (rate(rag_rate_limit_hits_total[5m]))
```
Type: Graph
Title: "Rate Limit Hits"

## Alerting

### Prometheus Alerts

**alerts.yml:**
```yaml
groups:
  - name: dartboard_rag
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: >
          sum(rate(rag_queries_total{status="error"}[5m])) /
          sum(rate(rag_queries_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Slow queries
      - alert: SlowQueries
        expr: >
          histogram_quantile(0.95,
            rate(rag_total_query_latency_seconds_bucket[5m])
          ) > 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Queries are slow"
          description: "95th percentile latency: {{ $value }}s"

      # Vector store empty
      - alert: VectorStoreEmpty
        expr: rag_vector_store_size == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Vector store is empty"
          description: "No documents in vector store"

      # High rate limit hits
      - alert: HighRateLimitHits
        expr: >
          rate(rag_rate_limit_hits_total[5m]) > 1
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Frequent rate limiting"
          description: "Rate limit hits: {{ $value }}/s"

      # Ingestion failures
      - alert: IngestionFailures
        expr: >
          sum(rate(rag_ingestions_total{status="error"}[1h])) /
          sum(rate(rag_ingestions_total[1h])) > 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "High ingestion failure rate"
          description: "{{ $value | humanizePercentage }} of ingestions failing"
```

**Load alerts into Prometheus:**
```bash
# Add to prometheus.yml
rule_files:
  - "alerts.yml"

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
```

### Grafana Alerts

Create alerts in Grafana dashboards:

1. Edit panel → Alert tab
2. Create alert rule
3. Set condition (e.g., "Query Time > 5s")
4. Configure notifications (email, Slack, PagerDuty)

## Custom Metrics

### Adding Application Metrics

```python
from prometheus_client import Counter, Histogram

# Define custom metric
custom_feature_usage = Counter(
    'rag_custom_feature_total',
    'Usage of custom feature',
    ['feature_name']
)

# Record metric
def use_feature(feature_name: str):
    custom_feature_usage.labels(feature_name=feature_name).inc()

    # Your feature code here
    ...
```

### Using Metric Decorators

```python
from dartboard.monitoring import (
    track_retrieval_time,
    track_generation_time,
    track_query_time
)

# Track retrieval performance
@track_retrieval_time
def custom_retrieval(query: str):
    # Automatically records retrieval latency
    return retriever.retrieve(query)

# Track generation performance
@track_generation_time
def custom_generation(query: str, chunks):
    # Automatically records generation latency
    return generator.generate(query, chunks)

# Track full query
@track_query_time(tier="premium")
async def handle_query(query: str):
    # Automatically records total time and increments counter
    chunks = custom_retrieval(query)
    result = custom_generation(query, chunks)
    return result
```

### Manual Metric Updates

```python
from dartboard.monitoring import (
    record_chunks_retrieved,
    record_chunks_created,
    update_vector_store_size,
    record_auth_attempt,
    record_rate_limit_hit
)

# After retrieval
chunks = retriever.retrieve(query, k=5)
record_chunks_retrieved(len(chunks))

# After ingestion
result = pipeline.ingest(document)
record_chunks_created(result.chunks_created)
update_vector_store_size(vector_store.count())

# After authentication
success = verify_credentials(api_key)
record_auth_attempt(success, tier="free")

# When rate limited
if rate_limited:
    record_rate_limit_hit(tier="free")
```

## Best Practices

### 1. Monitor Key SLAs

```promql
# Query success rate (target: >99%)
sum(rate(rag_queries_total{status="success"}[5m])) /
sum(rate(rag_queries_total[5m])) * 100

# Query latency P95 (target: <5s)
histogram_quantile(0.95, rate(rag_total_query_latency_seconds_bucket[5m]))

# System availability (target: >99.9%)
```

### 2. Set Up Alerts Early

- **Critical**: Vector store empty, high error rate
- **Warning**: Slow queries, ingestion failures
- **Info**: Rate limit hits, auth failures

### 3. Track Business Metrics

```python
# Custom business metrics
documents_by_category = Counter(
    'rag_documents_by_category',
    'Documents indexed by category',
    ['category']
)

user_queries_by_topic = Counter(
    'rag_queries_by_topic',
    'Queries by topic',
    ['topic']
)
```

### 4. Use Meaningful Labels

```python
# ✅ Good: Specific labels
query_counter.labels(status="success", tier="premium").inc()

# ❌ Bad: Generic labels
query_counter.labels(status="ok", tier="user").inc()
```

### 5. Aggregate Over Time Windows

```promql
# Use rate() for counters
rate(rag_queries_total[5m])

# Use appropriate time windows
# - Short term: 5m, 15m
# - Medium term: 1h, 6h
# - Long term: 1d, 7d
```

## Troubleshooting

### Issue: Metrics not appearing

**Check metrics endpoint:**
```bash
curl http://localhost:8000/metrics | grep rag_
```

**Verify Prometheus scraping:**
```bash
# Check targets in Prometheus UI
open http://localhost:9090/targets
```

### Issue: Incorrect metric values

**Verify metric type:**
- Counter: Always increases
- Gauge: Can go up/down
- Histogram: Buckets + sum + count

**Check query syntax:**
```promql
# Counter: Use rate()
rate(rag_queries_total[5m])

# Gauge: Use directly
rag_vector_store_size

# Histogram: Use histogram_quantile()
histogram_quantile(0.95, rate(rag_retrieval_latency_seconds_bucket[5m]))
```

### Issue: High cardinality warnings

**Avoid high-cardinality labels:**
```python
# ❌ Bad: User ID as label (millions of values)
metric.labels(user_id=user_id).inc()

# ✅ Good: User tier as label (few values)
metric.labels(tier=tier).inc()
```

## Example Queries

### Performance Analysis

```promql
# Average query time trend
rate(rag_total_query_latency_seconds_sum[5m]) /
rate(rag_total_query_latency_seconds_count[5m])

# Percentage of fast queries (<1s)
sum(rate(rag_total_query_latency_seconds_bucket{le="1"}[5m])) /
sum(rate(rag_total_query_latency_seconds_count[5m])) * 100

# Retrieval vs generation time ratio
rate(rag_retrieval_latency_seconds_sum[5m]) /
rate(rag_generation_latency_seconds_sum[5m])
```

### Capacity Planning

```promql
# Query rate trend
rate(rag_queries_total[1h])

# Storage growth rate
deriv(rag_vector_store_size[1h])

# Ingestion throughput
rate(rag_chunks_created_total[1h]) * 3600
```

### Error Investigation

```promql
# Top error types
topk(5, sum by (error_type) (rag_query_errors_total))

# Error rate by tier
sum by (tier) (rate(rag_queries_total{status="error"}[5m]))

# Ingestion failures by file type
sum by (file_type) (rag_ingestions_total{status="error"})
```

## See Also

- [API Guide](./api-guide.md) - API endpoints and usage
- [Generation](./generation.md) - LLM integration details
- [Retrieval Methods](./retrieval-methods.md) - Retrieval performance
- [Vector Stores](./vector-stores.md) - Storage metrics
