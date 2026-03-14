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
    
    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response is empty."
        )

    tool_calls = response.output

    if not tool_calls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Tool calls response is empty."
        )
        
    for tool_call in tool_calls:
        function_name = tool_call.name
        function_args = json.loads(tool_call.arguments)

        if function_name == "get_current_weather_from_owm":
            result = get_current_weather_from_owm(
                function_args.get("location"), function_args.get("unit", "metric")
            )
            messages.append(
                {
                    "toll_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "output": str(result),
                }
            )

    enriched_response = await llm_client.responses.create(
        input=messages,
        model=settings.llm.MODEL,
        max_output_tokens=max_tokens,
    )

    return enriched_response.output_text
