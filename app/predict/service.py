from fastapi.responses import StreamingResponse

from app.providers.protocol import LLMClient
from app.providers.factory import (
    get_inference_callable,
    get_stream_callable,
    get_weather_callable,
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
