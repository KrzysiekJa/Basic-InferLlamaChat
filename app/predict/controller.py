from fastapi import APIRouter, Request, Depends, status, HTTPException
from openai import AsyncOpenAI

from app.rate_limiting import limiter
from app.predict import deps
from app.predict.schemas import ChatInput, WeatherInput
from app.predict.service import (
    get_chat_inference_batch,
    get_chat_inference_stream,
    get_chat_inference_weather,
)
from app.logger import logger


router = APIRouter()


@router.post("/batch", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("6/minute")
async def run_chat_inference_batch(
    request: Request,
    chat_input: ChatInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    try:
        model_response = await get_chat_inference_batch(
            chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
        )
    except HTTPException as exc:
        logger.error(f"Error occured during batch inference: {exc.detail}")
        raise exc

    return model_response


@router.post("/stream", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("6/minute")
async def run_chat_inference_stream(
    request: Request,
    chat_input: ChatInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    return await get_chat_inference_stream(
        chat_input.user_prompt, chat_input.max_tokens, llm_client=llm_client
    )


@router.post("/weather", status_code=status.HTTP_200_OK, response_model=str)
@limiter.limit("4/minute")
async def run_chat_inference_weather(
    request: Request,
    weather_input: WeatherInput,
    llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client),
):
    try:
        model_response =  await get_chat_inference_weather(
            weather_input.user_prompt,
            weather_input.max_tokens,
            llm_client=llm_client,
        )
    except HTTPException as exc:
        logger.error(f"Error occured during weather inference: {exc.detail}")
        raise exc

    return model_response
