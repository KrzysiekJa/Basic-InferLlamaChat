from fastapi import APIRouter, Depends, status
from openai import AsyncOpenAI

from app.inference import deps
from app.inference.model import ChatInput, WeatherInput
from app.inference.service import (
    get_chat_inference_batch,
    get_chat_inference_stream,
    get_chat_inference_weather,
)


infer_router = APIRouter(prefix="/inference", tags=["inference"])


@infer_router.post("/batch", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_batch(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    return await get_chat_inference_batch(
        chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
    )


@infer_router.post("/stream", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_stream(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    return await get_chat_inference_stream(
        chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
    )


@infer_router.post("/weather", status_code=status.HTTP_200_OK, response_model=str)
async def run_chat_inference_weather(
    weather_input: WeatherInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    return await get_chat_inference_weather(
        weather_input.user_prompt,
        weather_input.max_tokens,
        llm_client=llm_client,
    )
