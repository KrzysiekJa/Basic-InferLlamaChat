import json
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.predict import deps
from app.prompts import CUSTOM_SYSTEM_PROMPT, OWM_TOOL_SYSTEM_PROMPT
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM
from app.tools.functions import get_current_weather_from_owm


async def get_chat_inference_batch(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    response = await llm_client.responses.create(
        input=[
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
        max_output_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response is empty."
        )

    return response.output_text


async def get_chat_inference_stream(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    response = await llm_client.responses.stream(
        input=[
            {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        model=settings.llm.MODEL,
        max_output_tokens=max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    return StreamingResponse(
        deps.stream_generator(response), media_type="text/event-stream"
    )


async def get_chat_inference_weather(
    user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI | None = None
) -> str:
    tools = [GET_CURRENT_WEATHER_FROM_OWM]
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
    response_output = response.output

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

    return enriched_response.output_text
