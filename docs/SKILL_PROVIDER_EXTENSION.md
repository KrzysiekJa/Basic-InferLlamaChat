# Skill: Adding New LLM Providers

Guide for extending Basic-InferLlamaChat with new LLM providers (Google GenAI, Anthropic, etc.).

## Overview

This project uses a **Factory Pattern** to support multiple providers without code duplication. Adding a new provider requires:
1. Creating a provider module
2. Updating the factory
3. Adding configuration
4. Testing locally

No changes needed to service layer or controllers.

## Step 1: Create Provider Module

### Location
Create file: `app/providers/{provider_name}.py`

### Structure
Each provider module must export:

```python
# Inference functions
async def run_{provider_name}_inference_batch(user_prompt, max_tokens, llm_client)
async def run_{provider_name}_inference_stream(user_prompt, max_tokens, llm_client)
async def run_{provider_name}_inference_weather(user_prompt, max_tokens, llm_client)

# Stream generator
async def stream_generator_{provider_name}(stream)

# Tool definition class
class {ProviderName}ToolDefinition:
    tools: list  # Tool schemas formatted for this provider
```

### Reference Implementations

Use existing providers as templates:
- **[app/providers/chat.py](app/providers/chat.py)** - Chat Completions API (Together, OpenRouter, standard models)
- **[app/providers/responses.py](app/providers/responses.py)** - OpenAI Responses API (proprietary)

### Example: Chat Completions Pattern

```python
from openai import AsyncOpenAI
from app.tools.definitions import get_tool_definitions

class GoogleGenAIToolDefinition:
    tools = get_tool_definitions()  # Adapt schema if needed

async def run_google_inference_batch(user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI):
    # Call provider-specific endpoint via llm_client
    response = await llm_client.chat.completions.create(
        model=settings.llm.MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
        tools=GoogleGenAIToolDefinition.tools
    )
    return response.choices[0].message.content

async def run_google_inference_stream(user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI):
    stream = await llm_client.chat.completions.create(
        model=settings.llm.MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        stream=True
    )
    return stream_generator_google(stream)

async def stream_generator_google(stream):
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Weather tool endpoint follows same pattern with tool calling logic
async def run_google_inference_weather(user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI):
    # Similar to run_google_inference_batch with tool calling flow
    pass
```

## Step 2: Update Factory

**File**: [app/providers/factory.py](app/providers/factory.py)

Add import and factory selection logic:

```python
from app.providers import google

def get_inference_callable():
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google.run_google_inference_batch
    elif settings.llm.DEFAULT_PROVIDER == "openai":
        return responses.run_responses_inference_batch
    else:
        return chat.run_chat_inference_batch

def get_stream_callable():
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google.run_google_inference_stream
    # ... similar pattern for other providers

def get_stream_generator():
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google.stream_generator_google
    # ... similar pattern

def get_weather_callable():
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google.run_google_inference_weather
    # ... similar pattern

def get_tool_definition():
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google.GoogleGenAIToolDefinition()
    # ... similar pattern
```

## Step 3: Update Configuration

**File**: [app/config.py](app/config.py)

Add provider-specific API keys to `LLMSettings` class:

```python
class LLMSettings(BaseSettings):
    # ... existing fields ...
    GOOGLE_API_KEY: str
    GOOGLE_API_URL: str
    # ... other providers ...
```

Update the `@model_validator` method to handle the new provider:

```python
@model_validator(mode="after")
def set_api_key(self) -> "LLMSettings":
    match self.DEFAULT_PROVIDER:
        case "google":
            self.API_KEY = self.GOOGLE_API_KEY
            self.BASE_URL = self.GOOGLE_API_URL
        case "openai":
            self.API_KEY = self.OPENAI_API_KEY
            self.BASE_URL = self.OPENAI_API_URL
        # ... other providers ...
        case _:
            self.API_KEY = self.OPENAI_API_KEY
            self.BASE_URL = self.OPENAI_API_URL
    return self
```

## Step 4: Update Environment Configuration

**File**: `app/example.env`

Add new provider credentials:

```env
# Google GenAI
GOOGLE_API_KEY=your-google-api-key
GOOGLE_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

## Step 5: Test Locally

### 1. Set Environment
```bash
cp app/example.env app/.env
# Edit app/.env and add GOOGLE_API_KEY and set DEFAULT_PROVIDER=google
```

### 2. Run Application
```bash
PYTHONPATH=. python app/main.py
```

### 3. Test Endpoints

**Batch inference:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Hello", "max_tokens": 50}'
```

**Streaming:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/stream \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Count to 5", "max_tokens": 50}'
```

**Weather tool:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/weather \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "What is the weather in London?", "max_tokens": 100}'
```

### 4. Verify Logs
Check `logs/` directory for any errors. Application logs via [app/logger.py](app/logger.py).

## Common Issues

| Issue | Solution |
|-------|----------|
| API key not recognized | Verify `DEFAULT_PROVIDER` matches case in factory and config validator |
| Tool calling fails | Ensure tool schema format matches provider specification (Chat Completions vs custom) |
| Stream response incomplete | Check async generator yields all chunks (see `stream_generator_*` pattern) |
| Rate limit exceeded | Endpoints have limits (batch: 6/min, weather: 4/min) - adjust in [app/rate_limiting.py](app/rate_limiting.py) |

## Checklist

- [ ] Provider module created with all required functions
- [ ] Tool definition class implemented
- [ ] Factory functions updated for all 5 callables
- [ ] Config.py has new API key fields
- [ ] Validator case handles new provider
- [ ] example.env updated with new credentials
- [ ] .env configured with real API key and DEFAULT_PROVIDER set
- [ ] App starts without errors
- [ ] All three endpoints (/batch, /stream, /weather) return responses
- [ ] Streaming responses show incremental output
