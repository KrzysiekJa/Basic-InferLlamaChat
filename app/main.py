import json
from pathlib import Path
from typing import AsyncGenerator

import requests
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from openai import AsyncOpenAI

from app.schemas import ChatInput
from app import deps
from app.config import settings


BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


app = FastAPI(title="Llama4Infer ChatApp")
api_router = APIRouter()


@api_router.get("/", status_code=200)
async def root():
    return "Navigate to /docs to see the API documentation."


@api_router.get("/ui", status_code=200)
async def ui(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


@api_router.post("/inference/batch", status_code=200, response_model=str)
async def run_chat_inference_batch(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    chat_completion = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
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
        raise HTTPException(status_code=404, detail="Chat completion is empty.")

    tokens = chat_completion.choices[0].message.content.split()
    return " ".join(tokens[: settings.CHAT_OUTPUT_MAX_TOKENS]).strip('"')


async def stream_generator(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    tokens_count = 0

    async for chunk in response:
        if not chunk.choices:
            break

        content = chunk.choices[0].delta.content
        tokens_count += len(content.split())

        if settings.CHAT_OUTPUT_MAX_TOKENS <= tokens_count:
            break

        yield content


@api_router.post("/inference/stream", status_code=200, response_model=str)
async def run_chat_inference_stream(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    response = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
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

    return StreamingResponse(stream_generator(response), media_type="text/event-stream")


def get_current_weather_from_owm(location: str, unit_sys: str = "metric") -> str:
    base_url, api_key = settings.weather_api.BASE_URL, settings.weather_api.OWM_API_KEY
    url = f"{base_url}q={location}&appid={api_key}&units={unit_sys}"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return json.dumps(
            {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "pressure_unit": "hPa" if unit_sys == "metric" else "inHg",
                "feels_like": data["main"]["feels_like"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"],
                "temperature_unit": "°C"
                if unit_sys == "metric"
                else "°F"
                if unit_sys == "imperial"
                else "K",
                "speed_unit": "m/s" if unit_sys == "metric" else "mph",
            }
        )
    return json.dumps(
        {
            "location": location,
            "temperature": "unknown",
            "description": "unknown",
        }
    )


@api_router.post("/inference/weather", status_code=200, response_model=str)
async def run_chat_inference_weather(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    tools = [
        {
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
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can access external functions. \
                    The responses from these functions will be appended to this dialog. \
                    Please, provide responses based on the information from these function calls.\
                    Your response should be short but concise, up to point. Try for complete phrases.",
        },
        {"role": "user", "content": chat_input.user_prompt},
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
        max_completion_tokens=chat_input.max_tokens,
    )

    tokens = enriched_response.choices[0].message.content.split()
    return " ".join(tokens[: settings.CHAT_OUTPUT_MAX_TOKENS]).strip('"')


app.include_router(api_router)


if "__main__" == __name__:
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8005, reload=True, log_level="debug"
    )
