# Skill: Deployment & Containerization

Guide for containerizing Basic-InferLlamaChat and setting up Redis caching for production environments.

## Overview

This skill covers:
1. Creating Docker multi-stage builds
2. Environment configuration for containers
3. Redis setup and integration
4. Health check endpoints
5. Production vs development settings

## Step 1: Create Dockerfile

**File**: `Dockerfile` (in project root)

Multi-stage build with dependencies and runtime separation:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Build dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --no-dev

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app/ ./app/
COPY docs/ ./docs/

# Set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app/main.py"]
```

## Step 2: Create Docker Compose

**File**: `docker-compose.yml`

Sets up FastAPI service with Redis caching:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: basic-inferllamachat
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
      - REDIS_URL=redis://redis:6379/0
      # Load .env file or set individual variables
      - DEFAULT_PROVIDER=${DEFAULT_PROVIDER:-openai}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OWM_API_KEY=${OWM_API_KEY}
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: basic-inferllamachat-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

## Step 3: Add Health Check Endpoint

**File**: [app/main.py](app/main.py)

Add a simple health check endpoint:

```python
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "basic-inferllamachat"}
```

This endpoint is used by Docker HEALTHCHECK and Kubernetes probes.

## Step 4: Update Environment Configuration

**File**: `app/config.py`

Add Redis URL configuration:

```python
class LLMSettings(BaseSettings):
    # ... existing fields ...
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # Cache TTL in seconds (1 hour default)
    
    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"
```

**File**: `app/example.env`

Add Redis configuration:

```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
```

## Step 5: Implement Redis Caching

### Create Cache Module

**File**: `app/cache.py`

```python
import hashlib
import json
from typing import Optional, Any
import aioredis
from app.config import settings
from app.logger import logger

redis_client: Optional[aioredis.Redis] = None

async def init_cache():
    """Initialize Redis connection."""
    global redis_client
    try:
        redis_client = await aioredis.create_redis_pool(settings.llm.REDIS_URL)
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")

async def close_cache():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        redis_client.close()
        await redis_client.wait_closed()
        logger.info("Redis cache closed")

def _make_cache_key(prefix: str, prompt: str, max_tokens: int) -> str:
    """Generate cache key from prompt and parameters."""
    key_data = f"{prefix}:{prompt}:{max_tokens}"
    hash_key = hashlib.md5(key_data.encode()).hexdigest()
    return f"inference:{prefix}:{hash_key}"

async def get_cached_response(prefix: str, prompt: str, max_tokens: int) -> Optional[str]:
    """Get cached inference response."""
    if not redis_client:
        return None
    
    try:
        key = _make_cache_key(prefix, prompt, max_tokens)
        cached = await redis_client.get(key)
        if cached:
            logger.debug(f"Cache hit: {key}")
            return cached.decode()
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
    
    return None

async def cache_response(prefix: str, prompt: str, max_tokens: int, response: str) -> None:
    """Cache inference response."""
    if not redis_client:
        return
    
    try:
        key = _make_cache_key(prefix, prompt, max_tokens)
        await redis_client.setex(key, settings.llm.CACHE_TTL, response)
        logger.debug(f"Cached response: {key}")
    except Exception as e:
        logger.error(f"Cache write error: {e}")
```

### Update Service Layer

**File**: [app/predict/service.py](app/predict/service.py)

Add caching to batch inference:

```python
from app.cache import get_cached_response, cache_response

async def get_inference_batch(user_prompt, max_tokens, llm_client):
    # Check cache first
    cached = await get_cached_response("batch", user_prompt, max_tokens)
    if cached:
        return cached
    
    # Get inference from provider
    inference_callable = providers.factory.get_inference_callable()
    model_response = await inference_callable(user_prompt, max_tokens, llm_client)
    
    # Cache the response
    await cache_response("batch", user_prompt, max_tokens, model_response)
    
    return model_response
```

### Update Lifespan

**File**: [app/main.py](app/main.py)

Initialize and close cache:

```python
from app.cache import init_cache, close_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_cache()
    yield
    # Shutdown
    await close_cache()
    limiter.try_acquire = lambda *args, **kwargs: True
```

## Step 6: Build and Run

### Development with Docker Compose

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Test health check
curl http://localhost:8000/health

# Stop services
docker-compose down
```

### Production Deployment

```bash
# Build image with tag
docker build -t basic-inferllamachat:latest .

# Push to registry (if applicable)
docker tag basic-inferllamachat:latest myregistry/basic-inferllamachat:latest
docker push myregistry/basic-inferllamachat:latest

# Run with explicit environment
docker run -d \
  --name llm-app \
  -p 8000:8000 \
  -e DEFAULT_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-xxx \
  -e REDIS_URL=redis://redis-host:6379/0 \
  --restart unless-stopped \
  basic-inferllamachat:latest
```

## Step 7: Kubernetes Deployment (Optional)

**File**: `k8s-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: basic-inferllamachat
spec:
  replicas: 2
  selector:
    matchLabels:
      app: basic-inferllamachat
  template:
    metadata:
      labels:
        app: basic-inferllamachat
    spec:
      containers:
      - name: app
        image: basic-inferllamachat:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEFAULT_PROVIDER
          value: "openai"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-key
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

## Configuration Summary

| Environment | Docker Compose | Kubernetes | Single Container |
|-------------|---|---|---|
| Redis | Service: `redis` | External/ClusterIP | None (disable caching) |
| API Keys | `.env` file | Secrets | Environment vars |
| Logs | Volume mount | Sidecar/ELK | stdout |
| Restart Policy | `unless-stopped` | Deployment controller | `--restart` |
| Health Check | HEALTHCHECK | livenessProbe | External monitoring |

## Checklist

- [ ] Dockerfile created with multi-stage build
- [ ] docker-compose.yml configured with app and Redis services
- [ ] Health check endpoint added to app/main.py
- [ ] Redis URL in config.py and example.env
- [ ] Cache module created (app/cache.py)
- [ ] Service layer updated with caching logic
- [ ] Lifespan updated to init/close cache
- [ ] docker-compose build succeeds
- [ ] docker-compose up starts both services
- [ ] Health check returns 200 OK
- [ ] API endpoints work through container
- [ ] Redis cache stores and retrieves values
- [ ] Logs directory mounts properly
