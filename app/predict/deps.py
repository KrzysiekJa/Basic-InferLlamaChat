from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.config import settings
from app.logger import logger
from app.providers.factory import get_stream_provider


async def get_llm_client() -> AsyncGenerator[AsyncOpenAI, None]:
    """Dependency to provide LLM client with correct configuration."""
    client = AsyncOpenAI(api_key=settings.llm.API_KEY, base_url=settings.llm.BASE_URL)
    logger.info("LLM client initialized successfully.")
    yield client


async def stream_generator(response: AsyncGenerator) -> AsyncGenerator[str, None]:
    """
    Automatically select the correct stream generator based on DEFAULT_PROVIDER.
    
    This function wraps the provider-specific stream generator and ensures
    the correct implementation is used without the caller knowing which provider is active.
    """
    provider = get_stream_provider()
    async for chunk in provider.stream_generator(response):
        yield chunk

