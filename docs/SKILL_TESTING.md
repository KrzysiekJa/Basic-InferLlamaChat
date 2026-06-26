# Skill: Testing & Verification

Guide for testing Basic-InferLlamaChat endpoints, providers, and features. Use this when validating changes, adding new providers, or before commits.

## Overview

This skill covers:
1. Manual API testing patterns
2. Provider-specific test scenarios
3. Rate limiting verification
4. Streaming response validation
5. Tool integration testing (weather)

## Prerequisites

Application running:
```bash
PYTHONPATH=. python app/main.py
```

## Test Data

Use these prompts consistently for reproducible testing:

```json
{
  "basic_prompt": "Say 'Hello from LLM'",
  "streaming_prompt": "Count from 1 to 5",
  "weather_prompt": "What is the weather in London?",
  "tool_use_prompt": "What's the current temperature in Paris?",
  "long_response": "Tell me a 200-word story about a robot"
}
```

## Section 1: Batch Inference Testing

### Endpoint
```
POST /api/v1/predict/batch
Content-Type: application/json
```

### Test 1.1: Basic Batch Request
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Say '\''Hello from LLM'\''",
    "max_tokens": 50
  }'
```

**Expected**: Response with text output, no streaming, completes immediately.

### Test 1.2: Maximum Tokens
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Tell me a 200-word story about a robot",
    "max_tokens": 200
  }'
```

**Expected**: Response approaches max_tokens limit, completes fully.

### Test 1.3: Invalid Request (Missing max_tokens)
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Hello"}'
```

**Expected**: 422 Unprocessable Entity with validation error.

### Test 1.4: Rate Limit (6/minute)
```bash
# Run 7 times rapidly
for i in {1..7}; do
  curl -s -X POST http://localhost:8000/api/v1/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"user_prompt": "test", "max_tokens": 10}' \
    -w "\n[Request $i] Status: %{http_code}\n"
done
```

**Expected**: First 6 succeed (200), 7th fails (429 Too Many Requests).

## Section 2: Streaming Inference Testing

### Endpoint
```
POST /api/v1/predict/stream
Content-Type: application/json
```

### Test 2.1: Basic Stream
```bash
curl -X POST http://localhost:8000/api/v1/predict/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Count from 1 to 5",
    "max_tokens": 50
  }' -N
```

**Expected**: Output appears incrementally (not all at once), readable stream chunks.

### Test 2.2: Stream Chunking
```bash
# Capture with timestamps to verify incremental output
curl -X POST http://localhost:8000/api/v1/predict/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Count from 1 to 10",
    "max_tokens": 100
  }' | while IFS= read -r line; do
    echo "$(date +%s%N): $line"
  done
```

**Expected**: Multiple lines with different timestamps (verifies streaming vs buffering).

### Test 2.3: Stream Interruption
```bash
timeout 2s curl -X POST http://localhost:8000/api/v1/predict/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Tell me a very long story",
    "max_tokens": 500
  }'
```

**Expected**: Connection closes cleanly after 2 seconds (no server errors in logs).

## Section 3: Weather Tool Testing

### Endpoint
```
POST /api/v1/predict/weather
Content-Type: application/json
```

**Prerequisite**: Set `OWM_API_KEY` in `.env` file.

### Test 3.1: Weather Query
```bash
curl -X POST http://localhost:8000/api/v1/predict/weather \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "What is the weather in London?",
    "max_tokens": 100
  }'
```

**Expected**: Response includes weather information or tool call result.

### Test 3.2: Multiple Location Query
```bash
curl -X POST http://localhost:8000/api/v1/predict/weather \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Compare weather in Paris, Berlin, and Amsterdam",
    "max_tokens": 200
  }'
```

**Expected**: Response handles multiple tool calls or locations.

### Test 3.3: Invalid Location
```bash
curl -X POST http://localhost:8000/api/v1/predict/weather \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "What'\''s the weather in XyzNotACity?",
    "max_tokens": 100
  }'
```

**Expected**: Graceful error handling (no 500 error, returns sensible response).

### Test 3.4: Rate Limit (4/minute)
```bash
# Run 5 times
for i in {1..5}; do
  curl -s -X POST http://localhost:8000/api/v1/predict/weather \
    -H "Content-Type: application/json" \
    -d '{"user_prompt": "weather", "max_tokens": 10}' \
    -w "[Request $i] Status: %{http_code}\n"
done
```

**Expected**: First 4 succeed (200), 5th fails (429).

## Section 4: Provider-Specific Testing

Test each configured provider to ensure compatibility.

### Test 4.1: Verify Current Provider
```bash
# Check logs to see which provider is being used
PYTHONPATH=. python app/main.py 2>&1 | grep -i provider
```

**Expected**: Log shows which provider is initialized (e.g., "Using OpenAI provider").

### Test 4.2: Switch Provider and Test

```bash
# Edit app/.env and change DEFAULT_PROVIDER
# Test batch, stream, and weather endpoints with new provider
```

Each provider should:
- ✅ Successfully complete batch requests
- ✅ Stream responses incrementally
- ✅ Handle weather tool calls
- ✅ Return valid responses for all 3 endpoints

### Test 4.3: Provider-Specific Features

| Provider | Test |
|----------|------|
| **OpenAI** | Test Responses API batch endpoint (if using responses.py) |
| **Together** | Verify streaming works with together.ai models |
| **OpenRouter** | Test model routing and fallback behavior |
| **Google GenAI** | Test Gemini-specific features (vision, extensions) |

## Section 5: Error Handling & Edge Cases

### Test 5.1: Empty Prompt
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "", "max_tokens": 50}'
```

**Expected**: Either valid response or 422 validation error (not 500).

### Test 5.2: Very Large max_tokens
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "test", "max_tokens": 100000}'
```

**Expected**: Request completes or provider error (not server crash).

### Test 5.3: Special Characters
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Unicode test: 你好 🚀 Ñoño",
    "max_tokens": 50
  }'
```

**Expected**: Response handles unicode without errors.

### Test 5.4: Missing API Key
```bash
# Temporarily unset API key in .env, restart app
unset OPENAI_API_KEY
PYTHONPATH=. python app/main.py
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "test", "max_tokens": 50}'
```

**Expected**: 500 error or clear error message (not silent failure).

## Section 6: Logging & Debugging

### View Application Logs

```bash
# Real-time logs
tail -f logs/default.log

# Search for errors
grep -i error logs/default.log

# Search for specific request
grep "user_prompt" logs/default.log
```

### Check Request Headers

App logs all requests. In [app/main.py](app/main.py) middleware:

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Headers: {dict(request.headers)}")
    # ...
```

### Monitor Rate Limiter

```bash
# Watch rate limit responses
while true; do
  curl -s http://localhost:8000/api/v1/predict/batch \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"user_prompt":"test","max_tokens":10}' \
    -w "Status: %{http_code}\n"
  sleep 5
done
```

## Testing Checklist

### Before Any Commit
- [ ] Batch endpoint returns valid response
- [ ] Stream endpoint streams incrementally
- [ ] Current provider works with all 3 endpoints
- [ ] No 500 errors in logs
- [ ] Rate limiting triggers correctly

### After Adding New Provider
- [ ] Provider module has all required functions
- [ ] Factory updated correctly
- [ ] Config/env updated with API key
- [ ] All 3 endpoints work with new provider
- [ ] Streaming is incremental (not buffered)
- [ ] Weather tool works with tool calling

### After Modifying Endpoints
- [ ] Response format matches schema
- [ ] Rate limits still apply
- [ ] Error responses have correct status codes
- [ ] Logging shows expected debug info

### After Docker/Deployment Changes
- [ ] Container starts without errors
- [ ] Health check returns 200
- [ ] Endpoints accessible on port 8000
- [ ] Logs appear in volume mount
- [ ] Redis cache operations work (if enabled)

## Quick Test Script

Save as `test_all.sh`:

```bash
#!/bin/bash

echo "=== Testing Batch ==="
curl -s -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt":"test","max_tokens":20}' | head -c 100

echo -e "\n=== Testing Stream ==="
curl -s -X POST http://localhost:8000/api/v1/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"user_prompt":"count to 3","max_tokens":20}' | head -c 100

echo -e "\n=== Testing Health ==="
curl -s http://localhost:8000/health | jq .

echo -e "\n=== Done ==="
```

Run:
```bash
chmod +x test_all.sh
./test_all.sh
```
