from typing import AsyncGenerator
from fastapi import Request
from openai import AsyncOpenAI

from app.config import settings
from app.logger import logger
from app.providers.factory import get_stream_generator
from app.providers.base import LLMClient


async def get_llm_client(request: Request) -> LLMClient:
    """Return the configured LLM client.

    For the google provider we expect a persistent client to be available on
    `request.app.state.google_genai_async` (initialized at application startup).
    """
    if settings.llm.DEFAULT_PROVIDER == "google":
        g_async = getattr(request.app.state, "google_genai_async", None)
        if g_async is None:
            # Fail fast with an informative error
            raise RuntimeError(
                "Google GenAI async client not initialized on app startup"
            )
        return g_async

    # Default: create an OpenAI Async client instance (no app-level persistence)
    client = AsyncOpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    logger.info("LLM client initialized successfully.")
    return client


async def stream_generator(response: AsyncGenerator) -> AsyncGenerator[str, None]:
    """
    Automatically select the correct stream generator based on DEFAULT_PROVIDER.

    This function wraps the provider-specific stream generator and ensures
    the correct implementation is used without the caller knowing which provider is active.
    """
    generator = get_stream_generator()
    async for chunk in generator(response):
        yield chunk
