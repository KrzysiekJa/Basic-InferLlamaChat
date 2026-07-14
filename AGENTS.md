# AI Agent Guide - Basic-InferLlamaChat

## Purpose
This repository is a FastAPI LLM inference app. Use this guide to understand the entrypoint, provider abstraction, request flow, and where agent changes should be made.

## What matters most
- App entrypoint: `app/main.py` with the FastAPI app exposed as `app`.
- Core inference routes: `app/predict/controller.py`.
- Provider selection: `app/providers/factory.py`, driven by `DEFAULT_PROVIDER` in `app/.env`.
- Business/service logic: `app/predict/service.py`.
- Config and environment validation: `app/config.py`.

## Quick Start Commands

### Setup (one-time)
```bash
pip install uv                           # Install package manager
uv venv .venv                            # Create virtual environment
source .venv/bin/activate                # Activate (Linux/macOS)
.venv\Scripts\activate                   # Activate (Windows)
uv sync --locked --all-extras            # Install dependencies
cp app/example.env app/.env              # Copy environment template
```

### Run Application
```bash
PYTHONPATH=. python app/main.py
```

**Environment Setup**: Copy `app/example.env` to `app/.env` and configure:
- API keys for your chosen provider (OPENAI_API_KEY, TOGETHER_API_KEY, or OPENROUTER_API_KEY)
- OpenWeatherMap API key (for weather tool feature)
- DEFAULT_PROVIDER setting to select which provider to use

## Architecture Overview

### Design Pattern: Provider Factory Pattern
The codebase implements **Factory Design Pattern** to support multiple LLM providers without code duplication. See [docs/PROVIDER_ARCHITECTURE.md](docs/PROVIDER_ARCHITECTURE.md) for detailed architecture.

**Key Components**:
1. **Base Abstractions** ([app/providers/base.py](app/providers/base.py)) - Protocol definitions for providers
2. **Provider Implementations**:
   - [app/providers/responses.py](app/providers/responses.py) - OpenAI Responses API
   - [app/providers/chat.py](app/providers/chat.py) - Chat Completions API (Together, OpenRouter, OpenAI)
   - [app/providers/google_genai.py](app/providers/google_genai.py) - Google GenAI implementation
3. **Factory** ([app/providers/factory.py](app/providers/factory.py)) - Selects correct provider at runtime
4. **Service Layer** ([app/predict/service.py](app/predict/service.py)) - Provider-agnostic business logic
5. **API Controller** ([app/predict/controller.py](app/predict/controller.py)) - REST endpoints

**Provider Selection**: Controlled via `DEFAULT_PROVIDER` setting in `.env`. Factory automatically loads the correct implementation—no code changes needed when switching providers.

## Directory Structure & Conventions

```
app/
├── main.py                 # FastAPI app initialization, lifespan, middleware, static mount
├── api.py                  # Route registration
├── config.py              # Pydantic settings (loaded from .env)
├── logger.py               # Logging initialization
├── rate_limiting.py        # Slowapi rate limiter configuration
├── predict/                # Core inference API
│   ├── controller.py       # FastAPI router endpoints (/batch, /stream, /weather)
│   ├── service.py          # Async service functions (provider-agnostic)
│   ├── deps.py             # Dependency injection (LLM client)
│   └── schemas.py          # Pydantic models (ChatInput, WeatherInput)
├── providers/              # Provider implementations (factory pattern)
│   ├── base.py             # Protocol definitions
│   ├── chat.py             # Chat Completions implementation
│   ├── responses.py        # OpenAI Responses API implementation
│   ├── google_genai.py    # Google GenAI provider implementation
│   ├── factory.py          # Provider selection logic
│   └── protocol.py         # Provider client protocol definitions
├── prompts/                # Jinja2 prompt templates
├── root/                   # Additional app-local utilities or placeholder content
├── static/                 # Static assets served at /static
│   ├── css/ui.css          # UI styles for rendered responses
│   └── js/ui.js            # Frontend markdown rendering and form handlers
├── templates/              # HTML templates (web UI)
│   └── index.html          # Main UI page for batch/stream/weather forms
├── tools/                  # Tool implementations
│   ├── definitions.py      # Tool schema definitions
│   └── functions.py         # Tool function handlers (weather, etc.)

docs/
├── PROVIDER_ARCHITECTURE.md  # Detailed architecture guide
├── SKILL_DEPLOYMENT.md       # Deployment notes
├── SKILL_PROVIDER_EXTENSION.md  # Provider extension notes
├── SKILL_TESTING.md          # Testing notes
logs/                        # Application logs directory
scripts/                     # Utility scripts
```

## Code Patterns & Conventions

### 1. Async/Await Throughout
- All service functions are `async def`
- Uses `AsyncOpenAI` client for non-blocking I/O
- Endpoints declared as `async def`

### 2. Dependency Injection
- FastAPI's `Depends()` for injecting LLM client and rate limiter
- See [app/predict/deps.py](app/predict/deps.py) for dependency definitions

### 3. Configuration via Pydantic
- [app/config.py](app/config.py) defines `LLMSettings` class
- Uses `pydantic_settings.BaseSettings` to load from `.env` file
- Provider-specific API keys configured via `@model_validator` method

### 4. Logging
- Centralized via [app/logger.py](app/logger.py) using `loguru`
- Call `init_logging()` in startup
- Use `logger.debug()`, `logger.error()` throughout

### 5. Rate Limiting
- Configured via Slowapi in [app/rate_limiting.py](app/rate_limiting.py)
- Applied to endpoints via `@limiter.limit()` decorator (e.g., "6/minute")
- Different limits per endpoint (batch: 6/min, weather: 4/min)

### 6. Type Hints
- All functions typed with return types and parameter types
- Pydantic models for request/response validation
- Uses Python 3.12+ syntax (e.g., `|` for unions)

## Testing & Validation

The project uses Pydantic schemas for request validation:
- [app/predict/schemas.py](app/predict/schemas.py) - Request/response models

No pytest configuration is currently configured. Manual testing via:
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Hello", "max_tokens": 100}'
```

## Common Development Tasks

### Adding a New Provider
1. Create provider module in `app/providers/` with inference functions
2. Update [app/providers/factory.py](app/providers/factory.py) with provider selection logic
3. Update [app/config.py](app/config.py) with provider-specific settings
4. No changes needed to service layer or controllers

### Modifying API Endpoints
1. Edit [app/predict/controller.py](app/predict/controller.py)
2. Update request/response schemas in [app/predict/schemas.py](app/predict/schemas.py)
3. Adjust rate limits via `@limiter.limit()` decorator if needed

### Adding New Tools
1. Define tool schema and function in [app/tools/definitions.py](app/tools/definitions.py)
2. Create handler in [app/tools/functions.py](app/tools/functions.py)
3. Update provider implementations to include new tool in tool calling logic

## Notes for agents
- Do not assume database, cache, or background job infrastructure exists.
- Respect `DEFAULT_PROVIDER` and use the provider factory instead of hardcoded provider branches.
- Keep UI changes separate from API behavior; the browser UI is a lightweight demo.
- Use existing Pydantic schemas and dependency injection instead of adding manual request parsing.

## References
- Provider architecture: [docs/PROVIDER_ARCHITECTURE.md](docs/PROVIDER_ARCHITECTURE.md)
- Testing notes: [docs/SKILL_TESTING.md](docs/SKILL_TESTING.md)
- Deployment notes: [docs/SKILL_DEPLOYMENT.md](docs/SKILL_DEPLOYMENT.md)
