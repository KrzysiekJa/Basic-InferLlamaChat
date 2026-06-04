# Provider Factory Pattern - Architecture Guide

## Overview

This codebase implements the **Factory Design Pattern** to seamlessly support multiple LLM providers (OpenAI, Together, OpenRouter, etc.). The pattern abstracts provider-specific implementations behind a unified interface, allowing you to switch providers by simply changing the `DEFAULT_PROVIDER` setting in the `.env` file.

The factory automatically selects the correct provider implementation at runtime based on configuration, with no code changes needed.

## Architecture Components

### 1. **Base Abstractions** (`app/providers/base.py`)

Uses Python Protocol (structural typing) to define the contract that all providers must implement:

```python
ToolDefinition  # Protocol for tool schema definitions
```

The Protocol approach provides flexible composition and better type checking support compared to inheritance-based Abstract Base Classes.

### 2. **Provider Implementations**

Each provider module exports callable functions for different inference modes and a ToolDefinition class:

#### **Responses API Provider** (`app/providers/responses.py`)
- **For**: OpenAI's proprietary Responses API
- **Functions**:
  - `run_responses_inference_batch()` - Batch inference
  - `run_responses_inference_stream()` - Streaming inference
  - `run_responses_inference_weather()` - Inference with weather tool support
  - `stream_generator_responses()` - Stream response generator
- **Classes**:
  - `ResponsesToolDefinition` - Tool schemas formatted for Responses API

#### **Chat Completions Provider** (`app/providers/chat.py`)
- **For**: Standard Chat Completions API (OpenAI Chat, Together, OpenRouter)
- **Functions**:
  - `run_chat_inference_batch()` - Batch inference
  - `run_chat_inference_stream()` - Streaming inference
  - `run_chat_inference_weather()` - Inference with weather tool support
  - `stream_generator_chat()` - Stream response generator
- **Classes**:
  - `ChatToolDefinition` - Tool schemas formatted for Chat Completions API

### 3. **Factory Pattern** (`app/providers/factory.py`)

The factory contains functions that return the correct provider callable based on configuration:

```python
get_inference_callable()      # Returns batch inference function
get_stream_callable()         # Returns streaming inference function
get_stream_generator()        # Returns stream generator function
get_weather_callable()        # Returns weather inference function
get_tool_definition()         # Returns tool definition instance
```

Selection logic (executed at runtime):

```python
if settings.llm.DEFAULT_PROVIDER == "openai":
    # Use OpenAI Responses API implementation
    return run_responses_inference_batch  # or other variants
else:
    # All other providers use Chat Completions API
    return run_chat_inference_batch  # or other variants
```

This approach provides clean separation of concerns and allows easy addition of new providers without modifying the service layer.

### 4. **Service Layer** (`app/predict/service.py`)

Provider-agnostic functions that automatically use the correct provider implementation:

```python
async def get_inference_batch(user_prompt, max_tokens, llm_client)
async def get_inference_stream(user_prompt, max_tokens, llm_client)
async def get_inference_weather(user_prompt, max_tokens, llm_client)
```

These functions:
1. Call the appropriate factory function to get the provider callable
2. Execute the callable with the provided arguments
3. Return the result (text for batch/weather, StreamingResponse for stream)

The factory selection happens transparently at runtime based on `DEFAULT_PROVIDER` setting.

### 5. **API Controller** (`app/predict/controller.py`)

Provides REST endpoints that handle HTTP requests and delegate to the service layer:

```python
@router.post("/batch")      # Batch inference endpoint
@router.post("/stream")     # Streaming inference endpoint
@router.post("/weather")    # Weather tool inference endpoint
```

Each endpoint:
1. Receives HTTP request with ChatInput or WeatherInput schema
2. Obtains LLM client from dependency injection (deps.get_llm_client)
3. Calls appropriate service function (get_inference_batch, get_inference_stream, get_inference_weather)
4. Returns response (text or streaming response)
5. Applies rate limiting based on configured limits

### 6. **Tool Definitions** (`app/tools/definitions.py`)

Provides provider-agnostic access to tool definitions:

```python
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM

# The factory automatically returns the correct format
# for the configured DEFAULT_PROVIDER
```

The tool definitions are automatically selected through `get_tool_definition()` from the factory, ensuring the correct schema format (Responses API vs Chat Completions) is used based on provider configuration.

---

## How to Use

### Basic Usage (No Code Changes Needed!)

Your existing code continues to work. The provider is automatically selected:

```python
from app.predict.service import get_inference_batch
from app.predict.deps import get_llm_client

# Provider selection happens automatically based on .env DEFAULT_PROVIDER
response = await get_inference_batch(prompt, max_tokens, llm_client)
```

### REST Endpoints

The API exposes three main endpoints at `/api/v1/predict/`:

```bash
# Batch inference (complete response)
POST /api/v1/predict/batch
Content-Type: application/json
{
  "user_prompt": "What is Python?",
  "max_tokens": 256
}

# Streaming inference (chunked response)
POST /api/v1/predict/stream
Content-Type: application/json
{
  "user_prompt": "Explain machine learning",
  "max_tokens": 256
}

# Inference with weather tool
POST /api/v1/predict/weather
Content-Type: application/json
{
  "user_prompt": "What's the weather like in New York?",
  "max_tokens": 256
}
```

### Switching Providers

Change `DEFAULT_PROVIDER` in `.env`:

```env
# Use OpenAI Responses API
DEFAULT_PROVIDER=openai

# or use Chat Completions (Together, OpenRouter, etc.)
DEFAULT_PROVIDER=together
DEFAULT_PROVIDER=openrouter
```

**No code changes needed** — the factory automatically selects the right implementation.

### Using Service Functions Directly

For programmatic access, use the provider-agnostic service functions:

```python
from app.predict.service import (
    get_inference_batch, 
    get_inference_stream, 
    get_inference_weather
)

# These work with any provider
batch_result = await get_inference_batch(prompt, 256, client)
stream_result = await get_inference_stream(prompt, 256, client)
weather_result = await get_inference_weather(prompt, 256, client)
```

---

## Adding a New Provider

To add support for a new provider or endpoint type:

### Step 1: Create Provider Module

Create a new file (e.g., `app/providers/my_provider.py`) with the following functions:

```python
# Inference functions
async def run_my_provider_inference_batch(
    user_prompt: str, 
    max_tokens: int, 
    llm_client: AsyncOpenAI | None = None
) -> str:
    """Implement batch inference for your provider."""
    # Your implementation here
    pass

async def run_my_provider_inference_stream(
    user_prompt: str, 
    max_tokens: int, 
    llm_client: AsyncOpenAI | None = None
) -> StreamingResponse:
    """Implement streaming inference."""
    pass

async def run_my_provider_inference_weather(
    user_prompt: str, 
    max_tokens: int, 
    llm_client: AsyncOpenAI | None = None
) -> str:
    """Implement weather tool inference."""
    pass

async def stream_generator_my_provider(response):
    """Implement stream response generator."""
    async for chunk in response:
        yield chunk

# Tool definition class
class MyProviderToolDefinition:
    @property
    def get_current_weather_from_owm(self) -> dict:
        """Return tool schema for your provider's format."""
        return {
            "type": "function",
            "function": {
                # Your tool definition
            }
        }
```

### Step 2: Update Factory

Add imports and update `app/providers/factory.py`:

```python
from app.providers.my_provider import (
    run_my_provider_inference_batch,
    run_my_provider_inference_stream,
    stream_generator_my_provider,
    run_my_provider_inference_weather,
    MyProviderToolDefinition,
)

def get_inference_callable() -> Callable:
    match settings.llm.DEFAULT_PROVIDER:
        case "openai":
            return run_responses_inference_batch
        case "my_provider":
            return run_my_provider_inference_batch
        case _:
            return run_chat_inference_batch

# Repeat for get_stream_callable(), get_stream_generator(), 
# get_weather_callable(), and get_tool_definition()
```

### Step 3: Configure Environment

Add to `.env`:

```env
DEFAULT_PROVIDER=my_provider
MY_PROVIDER_API_KEY=your_key_here
MY_PROVIDER_API_URL=https://api.myprovider.com
```

That's it! The service layer will automatically use your new provider without any further changes.

---

## Benefits of This Architecture

✅ **Provider Agnostic** - Switch providers with one config change in `.env`  
✅ **DRY Principle** - No duplicate conditional logic scattered throughout  
✅ **Type Safe** - Protocol-based typing provides static type checking  
✅ **Testable** - Easy to mock providers for unit testing  
✅ **Extensible** - Add new providers without modifying service layer or controller  
✅ **Single Responsibility** - Each module has one reason to change  
✅ **Clean Code** - Service layer and controller are provider-agnostic and simple  
✅ **Runtime Configuration** - Provider selected at runtime, no build-time coupling  
✅ **Flexible Composition** - Protocol-based design supports flexible composition patterns

---

## Project Structure

```
app/
├── api.py                 # Route registration
├── config.py              # Settings and environment configuration
├── logger.py              # Logging setup
├── main.py                # FastAPI app initialization
├── rate_limiting.py       # Rate limiting configuration
├── predict/
│   ├── controller.py      # REST API endpoints
│   ├── deps.py            # Dependency injection
│   ├── schemas.py         # Pydantic models (ChatInput, WeatherInput)
│   ├── service.py         # Provider-agnostic service functions
│   └── __init__.py
├── providers/
│   ├── base.py            # Protocol definitions
│   ├── chat.py            # Chat Completions implementation
│   ├── factory.py         # Factory pattern (provider selection)
│   ├── responses.py       # OpenAI Responses API implementation
│   └── __init__.py
├── tools/
│   ├── definitions.py     # Tool schema access
│   ├── functions.py       # Tool implementations
│   └── __init__.py
├── prompts/
│   ├── custom_system_prompt.md.j2
│   └── owm_tool_system_prompt.md.j2
├── templates/
│   └── index.html         # UI template
└── __init__.py
```

---

## Data Flow Diagram

```
HTTP Request
    ↓
Controller (predict/controller.py)
    ↓
Service Layer (predict/service.py)
    ├─ get_inference_batch()
    ├─ get_inference_stream()
    └─ get_inference_weather()
    ↓
Factory (providers/factory.py)
    ├─ get_inference_callable()
    ├─ get_stream_callable()
    ├─ get_stream_generator()
    └─ get_weather_callable()
    ↓
Provider Implementation
    ├─ app/providers/responses.py (if DEFAULT_PROVIDER=openai)
    └─ app/providers/chat.py (for all other providers)
    ↓
LLM API
    ├─ OpenAI API
    ├─ Together API
    └─ OpenRouter API
    ↓
HTTP Response
```

---

## Configuration

The application uses `pydantic-settings` for configuration management. Configuration is loaded from `.env` file in the `app/` directory:

```env
# LLM Settings
DEFAULT_PROVIDER=together          # Provider selection
MODEL=meta-llama/Llama-2-7b-hf    # Model identifier
CONTEXT_WINDOW=4096                # Max context window
MAX_TOKENS=256                     # Max generation tokens
TEMPERATURE=0.7                    # Sampling temperature

# API Keys and URLs
TOGETHER_API_KEY=your_key
TOGETHER_API_URL=https://api.together.xyz
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

Key setting for provider selection: `DEFAULT_PROVIDER`
- `openai` → Uses OpenAI Responses API
- `together` → Uses Chat Completions API with Together
- `openrouter` → Uses Chat Completions API with OpenRouter
- Any other value → Uses Chat Completions API as default

## Class Diagram

```
BaseInferenceProvider (ABC)
├── ResponsesInferenceProvider (OpenAI Responses API)
└── ChatInferenceProvider (Chat Completions API)

BaseStreamProvider (ABC)
├── ResponsesStreamProvider
└── ChatStreamProvider

BaseToolDefinition (ABC)
├── ResponsesToolDefinition
└── ChatToolDefinition

Factory Functions:
├── get_inference_provider() → BaseInferenceProvider
├── get_stream_provider() → BaseStreamProvider
└── get_tool_definition() → BaseToolDefinition
```

---

## File Structure

```
app/
├── providers/                    # Provider abstraction layer
│   ├── __init__.py
│   ├── base.py                  # Abstract base classes
│   ├── factory.py               # Provider selection factory
│   ├── responses.py             # OpenAI Responses API implementation
│   └── chat.py                  # Chat Completions API implementation
├── predict/
│   ├── service.py              # Uses factory for provider selection
│   ├── deps.py                 # Uses factory for stream providers
│   └── ...
├── tools/
│   ├── definitions.py           # Unified tool definitions (provider-agnostic)
│   ├── functions.py
│   └── ...
└── config.py                    # Contains DEFAULT_PROVIDER setting
```

---

## Migration Guide for Existing Code

### Old Way (Deprecated)
```python
# These files no longer exist, but the unified file works the same way
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM
from app.predict.service import get_responses_inference_batch  # Deprecated alias
```

### New Way (Recommended)
```python
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM
from app.predict.service import get_inference_batch, get_inference_stream
from app.predict.deps import get_llm_client, stream_generator
```

All functions now work with any provider through the factory pattern.


---

## Summary

The Repository Design Pattern provides a clean, extensible way to support multiple LLM providers. The factory automatically selects the correct implementation based on your `DEFAULT_PROVIDER` setting, making your code provider-agnostic and maintainable.

**Configuration-driven, not code-driven.** 🎯
