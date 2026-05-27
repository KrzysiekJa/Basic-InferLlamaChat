# Provider Factory Pattern - Architecture Guide

## Overview

This codebase implements the **Factory Design Pattern** to seamlessly support multiple LLM providers (OpenAI, Together, OpenRouter, etc.). The pattern abstracts provider-specific implementations behind a unified interface, allowing you to switch providers by simply changing the `DEFAULT_PROVIDER` setting in `config.py`.

## Architecture Components

### 1. **Base Abstractions** (`app/providers/base.py`)

Three abstract base classes define the contract that all providers must implement:

```python
BaseInferenceProvider  # Inference operations (batch, stream, weather tools)
BaseStreamProvider     # Stream generation
BaseToolDefinition     # Tool schema definitions
```

### 2. **Provider Implementations**

#### **Responses API Provider** (`app/providers/responses.py`)
- **For**: OpenAI's proprietary Responses API
- **Classes**: 
  - `ResponsesInferenceProvider` - Handles inference calls
  - `ResponsesStreamProvider` - Manages streaming responses
  - `ResponsesToolDefinition` - Tool schemas for Responses API

#### **Chat Completions Provider** (`app/providers/chat.py`)
- **For**: Standard OpenAI Chat API (and other providers like Together, OpenRouter)
- **Classes**:
  - `ChatInferenceProvider` - Handles inference calls
  - `ChatStreamProvider` - Manages streaming responses
  - `ChatToolDefinition` - Tool schemas for Chat API

### 3. **Factory Pattern** (`app/providers/factory.py`)

The factory contains three functions that automatically select the correct provider:

```python
get_inference_provider()    # Returns appropriate inference provider
get_stream_provider()       # Returns appropriate stream provider
get_tool_definitions()      # Returns appropriate tool definitions
```

The selection logic:
```python
if settings.llm.DEFAULT_PROVIDER == "openai":
    return ResponsesImplementation()  # Uses Responses API
else:
    return ChatImplementation()       # Uses Chat Completions API
```

### 4. **Updated Service Layer** (`app/predict/service.py`)

Provider-agnostic functions that automatically use the correct implementation:

```python
async def get_inference_batch(user_prompt, max_tokens, llm_client)
async def get_inference_stream(user_prompt, max_tokens, llm_client)
async def get_inference_weather(user_prompt, max_tokens, llm_client)
```

Plus backward-compatible aliases for existing code.

### 5. **Updated Dependencies** (`app/predict/deps.py`)

The stream generator is now provider-agnostic:

```python
async def stream_generator(response):
    provider = get_stream_provider()
    async for chunk in provider.stream_generator(response):
        yield chunk
```

### 6. **Tool Definitions** (`app/tools/definitions.py`)

Provider-agnostic tool definitions that automatically select the correct format:

```python
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM

# The factory automatically returns the correct format
# for the configured DEFAULT_PROVIDER
```

---

## How to Use

### Basic Usage (No Code Changes Needed!)

Your existing code continues to work. The provider is automatically selected:

```python
from app.predict.service import get_inference_batch
from app.predict.deps import get_llm_client

# Provider selection happens automatically based on config
response = await get_inference_batch(prompt, max_tokens, llm_client)
```

### Switching Providers

Just change `DEFAULT_PROVIDER` in `.env`:

```env
# .env
DEFAULT_PROVIDER=openai        # Uses Responses API
# or
DEFAULT_PROVIDER=together      # Uses Chat Completions API
# or
DEFAULT_PROVIDER=openrouter    # Uses Chat Completions API
```

**No code changes needed** — the factory automatically selects the right implementation.

### New Recommended API

Use the provider-agnostic functions:

```python
from app.predict.service import get_inference_batch, get_inference_stream, get_inference_weather

# These work with any provider
batch_result = await get_inference_batch(prompt, 256, client)
stream_result = await get_inference_stream(prompt, 256, client)
weather_result = await get_inference_weather(prompt, 256, client)
```

---

## Adding a New Provider

To add support for a new provider or endpoint type:

### Step 1: Determine API Style
- **Similar to OpenAI Responses API?** → Extend `ResponsesInferenceProvider`
- **Uses standard Chat Completions?** → Extend `ChatInferenceProvider`
- **Completely different?** → Create a new implementation file

### Step 2: Create Implementation Classes

```python
# app/providers/my_provider.py
from app.providers.base import BaseInferenceProvider, BaseStreamProvider, BaseToolDefinition

class MyProviderInference(BaseInferenceProvider):
    async def inference_batch(self, user_prompt, max_tokens, llm_client):
        # Your implementation
        pass
    
    async def inference_stream(self, user_prompt, max_tokens, llm_client):
        # Your implementation
        pass
    
    async def inference_weather(self, user_prompt, max_tokens, llm_client):
        # Your implementation
        pass

class MyProviderStream(BaseStreamProvider):
    async def stream_generator(self, response):
        # Your stream implementation
        pass

class MyProviderToolDef(BaseToolDefinition):
    @property
    def get_current_weather_from_owm(self):
        # Your tool schema
        pass
```

### Step 3: Update Factory

```python
# app/providers/factory.py
def get_inference_provider():
    match settings.llm.DEFAULT_PROVIDER:
        case "openai":
            return ResponsesInferenceProvider()
        case "my_provider":
            return MyProviderInference()
        case _:
            return ChatInferenceProvider()
```

---

## Benefits of This Architecture

✅ **Provider Agnostic** - Switch providers with one config change  
✅ **DRY Principle** - No duplicate conditional logic scattered throughout  
✅ **Type Safe** - Abstract base classes enforce contracts  
✅ **Testable** - Easy to mock providers for testing  
✅ **Extensible** - Add new providers without modifying existing code  
✅ **Backward Compatible** - Old function names still work  
✅ **Single Responsibility** - Each class has one reason to change  
✅ **Clean Code** - Service layer is provider-agnostic and simple  

---

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
└── get_tool_definitions() → BaseToolDefinition
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
