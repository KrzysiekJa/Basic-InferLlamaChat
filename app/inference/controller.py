import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.inference import deps
from app.inference.prompts import CUSTOM_SYSTEM_PROMPT, OWM_TOOL_SYSTEM_PROMPT
from app.inference.model import ChatInput, WeatherInput
from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM
from app.tools.functions import get_current_weather_from_owm


infer_router = APIRouter(prefix="/inference", tags=["inference"])


@infer_router.post("/batch", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_batch(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    chat_completion = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": chat_input.user_prompt,
            },
        ],
        model=settings.llm.MODEL,
        max_completion_tokens=chat_input.max_tokens,
        temperature=settings.llm.TEMPERATURE,
    )

    if not chat_completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat completion is empty."
        )

    tokens = chat_completion.choices[0].message.content.split()
    return " ".join(tokens[: settings.CHAT_OUTPUT_MAX_TOKENS]).strip()


@infer_router.post("/stream", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_stream(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    response = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": chat_input.user_prompt,
            },
        ],
        model=settings.llm.MODEL,
        max_completion_tokens=chat_input.max_tokens,
        temperature=settings.llm.TEMPERATURE,
        stream=True,
    )

    return StreamingResponse(
        deps.stream_generator(response), media_type="text/event-stream"
    )


@infer_router.post("/weather", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_weather(
    weather_input: WeatherInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    tools = [GET_CURRENT_WEATHER_FROM_OWM]
    messages = [
        {
            "role": "system",
            "content": OWM_TOOL_SYSTEM_PROMPT,
        },
        {"role": "user", "content": weather_input.user_prompt},
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
        max_completion_tokens=weather_input.max_tokens,
    )

    tokens = enriched_response.choices[0].message.content.split()
    return " ".join(tokens[: settings.CHAT_OUTPUT_MAX_TOKENS]).strip()
