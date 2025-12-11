# Dartboard RAG API - Deployment Guide

Complete guide for deploying the Dartboard RAG API using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Monitoring](#monitoring)
- [Production Considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Docker Engine 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose 2.0+ (included with Docker Desktop)
- OpenAI API key ([Get API key](https://platform.openai.com/api-keys))

### Recommended
- 2+ CPU cores
- 4GB+ RAM
- 10GB+ disk space

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dartboard_rig.git
cd dartboard_rig

# Create environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional (defaults shown)
API_PORT=8000
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Build and Run

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Check health
curl http://localhost:8000/health
```

### 4. Verify Deployment

```bash
# Test health endpoint
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs

# Check metrics
curl http://localhost:8000/metrics
```

## Configuration

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | `sk-...` |

#### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `API_PORT` | Port to expose API | `8000` |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | Sentence transformer model | `sentence-transformers/all-MiniLM-L6-v2` |
| `VECTOR_STORE_TYPE` | Vector store backend | `faiss` |
| `VECTOR_STORE_PATH` | Path to store vectors | `/app/data/vector_store` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Volume Mounts

The API uses persistent volumes for data storage:

```yaml
volumes:
  - ./data:/app/data  # Vector store and uploaded documents
```

## Deployment Options

### Option 1: API Only (Recommended for Getting Started)

```bash
# Start just the API service
docker-compose up -d api

# Access at http://localhost:8000
```

### Option 2: API + Monitoring Stack

```bash
# Start API with Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Option 3: Development Mode

For development with hot-reload:

```yaml
# Uncomment in docker-compose.yml:
volumes:
  - ./dartboard:/app/dartboard  # Mount source code
```

Then:

```bash
docker-compose up  # Run without -d to see logs
```

## Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090` (when using monitoring profile).

**Useful Queries:**

```promql
# Query latency (95th percentile)
histogram_quantile(0.95, rate(rag_total_query_latency_seconds_bucket[5m]))

# Error rate
rate(rag_query_errors_total[5m])

# Requests per tier
sum by (tier) (rate(rag_requests_by_tier_total[5m]))

# Vector store size
rag_vector_store_size
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin).

Create dashboards to visualize:
- Query latency and throughput
- Error rates by type
- Vector store growth
- Rate limiting statistics
- Authentication metrics

### Health Checks

Docker includes automated health checks:

```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:8000/health
```

## Production Considerations

### Security

1. **API Keys**
   - Use strong API keys
   - Rotate keys regularly
   - Never commit keys to version control

2. **Network Security**
   - Use HTTPS in production (add reverse proxy like nginx)
   - Restrict network access
   - Enable firewall rules

3. **Container Security**
   - API runs as non-root user (`appuser`)
   - Minimal base image (python:3.13-slim)
   - No unnecessary packages

### Performance

1. **Resource Limits**

```yaml
# Add to docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

2. **Scaling**

```bash
# Scale API horizontally
docker-compose up -d --scale api=3
```

3. **Caching**
   - Vector store is cached on disk
   - Consider using Redis for additional caching

### Backup

```bash
# Backup vector store data
docker run --rm -v dartboard_rig_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/vector-store-backup.tar.gz -C /data .

# Restore from backup
docker run --rm -v dartboard_rig_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/vector-store-backup.tar.gz -C /data
```

### Logging

1. **Application Logs**

```bash
# View logs
docker-compose logs -f api

# Export logs
docker-compose logs api > api-logs.txt
```

2. **Log Rotation**

Configure log rotation in `docker-compose.yml`:

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose logs api

# Check container status
docker ps -a

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### API Not Accessible

```bash
# Check if port is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Check container network
docker network inspect dartboard_rig_rag-network

# Test from within container
docker-compose exec api curl http://localhost:8000/health
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase memory limit in docker-compose.yml
# See "Resource Limits" section above
```

### OpenAI API Errors

```bash
# Verify API key is set
docker-compose exec api env | grep OPENAI

# Test OpenAI connection
docker-compose exec api python -c \
  "from openai import OpenAI; print(OpenAI().models.list())"
```

### Vector Store Issues

```bash
# Check data volume
docker volume inspect dartboard_rig_data

# Reset vector store (WARNING: deletes all data)
docker-compose down -v
rm -rf ./data/*
docker-compose up -d
```

## Useful Commands

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (deletes data)
docker-compose down -v

# View resource usage
docker stats

# Execute command in container
docker-compose exec api bash

# Rebuild specific service
docker-compose build --no-cache api

# View container logs (last 100 lines)
docker-compose logs --tail=100 api

# Follow logs in real-time
docker-compose logs -f api
```

## Next Steps

1. **Ingest Documents**: Upload documents via POST `/ingest`
2. **Query**: Ask questions via POST `/query`
3. **Monitor**: Check `/metrics` endpoint
4. **Scale**: Add more API replicas as needed
5. **Optimize**: Tune based on Prometheus metrics

## Support

For issues and questions:
- GitHub Issues: [github.com/yourusername/dartboard_rig/issues](https://github.com/yourusername/dartboard_rig/issues)
- Documentation: [github.com/yourusername/dartboard_rig/wiki](https://github.com/yourusername/dartboard_rig/wiki)

---

**Generated with [Claude Code](https://claude.com/claude-code)**
