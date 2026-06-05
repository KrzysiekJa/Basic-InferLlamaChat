"""OpenAI Responses API provider implementation."""

import json
import contextlib
from typing import AsyncGenerator
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.prompts import CUSTOM_SYSTEM_PROMPT, OWM_TOOL_SYSTEM_PROMPT


async def run_responses_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
) -> str:
    messages = [
        {
            "role": "system",
            "content": CUSTOM_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = await llm_client.responses.create(
        input=messages,
        model=settings.llm.MODEL,
        max_output_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response is empty."
        )

    return response.output_text


async def run_responses_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
) -> StreamingResponse:
    messages = [
        {
            "role": "system",
            "content": CUSTOM_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = await llm_client.responses.stream(
        input=messages,
        model=settings.llm.MODEL,
        max_output_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    return StreamingResponse(
        stream_generator_responses(response), media_type="text/event-stream"
    )


async def run_responses_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
) -> str:
    from app.providers.factory import get_tool_definition
    from app.tools.functions import get_current_weather_from_owm

    tool_defs = get_tool_definition()
    tools = [tool_defs.get_current_weather_from_owm]
    messages = [
        {
            "role": "system",
            "content": OWM_TOOL_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_prompt},
    ]

    response = await llm_client.responses.create(
        input=messages,
        model=settings.llm.MODEL,
        max_output_tokens=settings.weather_api.MAX_TOKENS,
        tools=tools,
        tool_choice="required",
    )
    response_output = response.output_text

    if not response_output:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response output is empty."
        )

    for item in response_output:
        item_name = item.name

        if item.type != "function_call" or item_name != "get_current_weather_from_owm":
            continue

        args = json.loads(item.arguments)
        result = get_current_weather_from_owm(
            args.get("location"), args.get("unit", "metric")
        )
        messages.append(
            {
                "call_id": item.call_id,
                "type": "function_call_output",
                "name": item_name,
                "output": str(result),
            }
        )

    enriched_response = await llm_client.responses.create(
        input=messages,
        model=settings.llm.MODEL,
        max_output_tokens=max_tokens,
    )

    if not enriched_response.output_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Enriched response is empty."
        )

    return enriched_response.output_text


async def stream_generator_responses(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    """Stream generator for responses API."""
    tokens_count = 0

    async with contextlib.aclosing(response) as resp:
        async for chunk in resp:
            if not chunk.type == "response.completed":
                break

            content = chunk.delta
            tokens_count += len(content.split())

            if settings.chat.OUTPUT_MAX_TOKENS <= tokens_count:
                break

            yield content


class ResponsesToolDefinition:
    """Tool definitions for OpenAI Responses API."""

    @property
    def get_current_weather_from_owm(self) -> dict:
        """Get tool definition for weather function (Responses API format)."""
        return {
            "type": "function",
            "name": "get_current_weather_from_owm",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city or state, e.g. San Francisco, CA",
                    },
                    "unit_sys": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["location"],
            },
        }
