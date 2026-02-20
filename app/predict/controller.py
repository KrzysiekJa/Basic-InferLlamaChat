from fastapi import APIRouter, Request, Depends, status
from openai import AsyncOpenAI

from app.rate_limiting import limiter
from app.predict import deps
from app.predict.model import ChatInput, WeatherInput
from app.predict.service import (
    get_chat_inference_batch,
    get_chat_inference_stream,
    get_chat_inference_weather,
)


predict_router = APIRouter(prefix="/predict", tags=["predict", "inference"])


@predict_router.post("/batch", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("5/minute")
async def run_chat_inference_batch(
    request: Request,
    chat_input: ChatInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    return await get_chat_inference_batch(
        chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
    )


@predict_router.post("/stream", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("5/minute")
async def run_chat_inference_stream(
    request: Request,
    chat_input: ChatInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    return await get_chat_inference_stream(
        chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
    )


@predict_router.post("/weather", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("3/minute")
async def run_chat_inference_weather(
    request: Request,
    weather_input: WeatherInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    return await get_chat_inference_weather(
        weather_input.user_prompt,
        weather_input.max_tokens,
        llm_client=llm_client,
    )
