from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse

from app.providers.factory import get_inference_provider, get_stream_provider


# --- Provider-agnostic inference functions ---
# These functions automatically select the right implementation based on DEFAULT_PROVIDER

async def get_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Perform batch inference using the configured provider."""
    provider = get_inference_provider()
    return await provider.inference_batch(user_prompt, max_tokens, llm_client)


async def get_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> StreamingResponse:
    """Perform streaming inference using the configured provider."""
    provider = get_inference_provider()
    return await provider.inference_stream(user_prompt, max_tokens, llm_client)


async def get_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Perform inference with weather tool using the configured provider."""
    provider = get_inference_provider()
    return await provider.inference_weather(user_prompt, max_tokens, llm_client)


# --- Backward compatibility aliases ---
# Keep the old function names for existing code

async def get_responses_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Deprecated: Use get_inference_batch instead."""
    return await get_inference_batch(user_prompt, max_tokens, llm_client)


async def get_responses_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> StreamingResponse:
    """Deprecated: Use get_inference_stream instead."""
    return await get_inference_stream(user_prompt, max_tokens, llm_client)


async def get_responses_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Deprecated: Use get_inference_weather instead."""
    return await get_inference_weather(user_prompt, max_tokens, llm_client)


async def get_chat_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Deprecated: Use get_inference_batch instead."""
    return await get_inference_batch(user_prompt, max_tokens, llm_client)


async def get_chat_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> StreamingResponse:
    """Deprecated: Use get_inference_stream instead."""
    return await get_inference_stream(user_prompt, max_tokens, llm_client)


async def get_chat_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    """Deprecated: Use get_inference_weather instead."""
    return await get_inference_weather(user_prompt, max_tokens, llm_client)
