from fastapi.responses import StreamingResponse

from app.providers.base import LLMClient
from app.providers.factory import (
    get_inference_callable,
    get_stream_callable,
    get_weather_callable,
    get_calculator_callable,
)


# --- Provider-agnostic inference functions ---
# These functions automatically select the right implementation based on DEFAULT_PROVIDER


async def get_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: LLMClient | None = None
) -> str:
    """Perform batch inference using the configured provider."""
    func = get_inference_callable()
    return await func(user_prompt, max_tokens, llm_client)


async def get_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: LLMClient | None = None
) -> StreamingResponse:
    """Perform streaming inference using the configured provider."""
    func = get_stream_callable()
    return await func(user_prompt, max_tokens, llm_client)


async def get_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: LLMClient | None = None
) -> str:
    """Perform inference with weather tool using the configured provider."""
    func = get_weather_callable()
    return await func(user_prompt, max_tokens, llm_client)


async def get_inference_calculator(
    user_prompt: str, max_tokens: int, llm_client: LLMClient | None = None
) -> str:
    """Perform inference with calculator tool using the configured provider."""
    func = get_calculator_callable()
    return await func(user_prompt, max_tokens, llm_client)
