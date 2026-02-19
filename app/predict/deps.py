from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.config import settings


async def get_llm_client() -> AsyncGenerator[AsyncOpenAI, None]:
    client = AsyncOpenAI(
        api_key=settings.llm.TOGETHER_API_KEY, base_url=settings.llm.BASE_URL
    )
    yield client


async def stream_generator(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    tokens_count = 0

    async for chunk in response:
        if not chunk.choices:
            break

        content = chunk.choices[0].delta.content
        tokens_count += len(content.split())

        if settings.chat.OUTPUT_MAX_TOKENS <= tokens_count:
            break

        yield content
