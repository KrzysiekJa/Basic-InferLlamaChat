import json
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.inference import deps
from app.prompts import CUSTOM_SYSTEM_PROMPT, OWM_TOOL_SYSTEM_PROMPT
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM
from app.tools.functions import get_current_weather_from_owm


async def get_chat_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
):
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
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat completion is empty."
        )

    return chat_completion.choices[0].message.content


async def get_chat_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
):
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
        deps.stream_generator(response), media_type="text/event-stream"
    )


async def get_chat_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
):
    tools = [GET_CURRENT_WEATHER_FROM_OWM]
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

    if tool_calls:
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

    return enriched_response.choices[0].message.content
