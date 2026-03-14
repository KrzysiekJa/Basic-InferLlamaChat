from typing import AsyncGenerator
from openai import AsyncOpenAI
import contextlib

from app.config import settings
from app.logger import logger


async def get_llm_client() -> AsyncGenerator[AsyncOpenAI, None]:
    client = AsyncOpenAI(
        api_key=settings.llm.OPENAI_API_KEY, base_url=settings.llm.BASE_URL
    )
    logger.info("LLM client initialized successfully.")
    yield client


async def stream_generator(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    tokens_count = 0

    async with contextlib.aclosing(response) as resp:
        # Added extra `with` block for safety reasons - for generator cleanup.
        # It awaits generator's `aclose` method on the way out.
        async for chunk in resp:
            if not chunk.type == "response.completed":
                break

            content = chunk.delta
            tokens_count += len(content.split())

            if settings.chat.OUTPUT_MAX_TOKENS <= tokens_count:
                break

            yield content
