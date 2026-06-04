"""Chat Completions API provider implementation."""

import json
import contextlib
from typing import AsyncGenerator
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.prompts import CUSTOM_SYSTEM_PROMPT, OWM_TOOL_SYSTEM_PROMPT


async def run_chat_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
) -> str:
    chat_completion = await llm_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": CUSTOM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        model=settings.llm.MODEL,
        max_completion_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    if not chat_completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response is empty."
        )

    return chat_completion.choices[0].message.content


async def run_chat_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
) -> StreamingResponse:
    response = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        model=settings.llm.MODEL,
        max_completion_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
        stream=True,
    )

    return StreamingResponse(
        stream_generator_chat(response), media_type="text/event-stream"
    )


async def run_chat_inference_weather(
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

    chat_completion = await llm_client.chat.completions.create(
        messages=messages,
        model=settings.llm.MODEL,
        max_completion_tokens=settings.weather_api.MAX_TOKENS,
        tools=tools,
        tool_choice="required",
    )
    tool_calls = chat_completion.choices[0].message.tool_calls

    if not tool_calls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response output is empty."
        )

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if function_name == "get_current_weather_from_owm":
            function_response = get_current_weather_from_owm(
                function_args.get("location"), function_args.get("unit", "metric")
            )
            messages.append(
                {
                    "toll_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    enriched_response = await llm_client.chat.completions.create(
        messages=messages,
        model=settings.llm.MODEL,
        max_completion_tokens=max_tokens,
    )

    if not enriched_response.choices[0].message.content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Enriched response is empty."
        )

    return enriched_response.choices[0].message.content


async def stream_generator_chat(response: AsyncGenerator) -> AsyncGenerator[str, None]:
    """Stream generator for chat completions API."""
    tokens_count = 0

    async with contextlib.aclosing(response) as resp:
        async for chunk in resp:
            if not chunk.choices:
                break

            content = chunk.choices[0].delta.content
            tokens_count += len(content.split())

            if settings.chat.OUTPUT_MAX_TOKENS <= tokens_count:
                break

            yield content


class ChatToolDefinition:
    """Tool definitions for Chat Completions API."""

    @property
    def get_current_weather_from_owm(self) -> dict:
        """Get tool definition for weather function (Chat API format)."""
        return {
            "type": "function",
            "function": {
                "name": "get_current_weather_from_owm",
                "description": "Get the current weather in a given location",
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
            },
        }
