from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from openai import AsyncOpenAI
from pydantic import BaseModel

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


class ChatInput(BaseModel):
    user_prompt: str = "Tell me about Nicolas Cage."
    max_tokens: int = settings.llm.MAX_TOKENS


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
        raise HTTPException(status_code=404, detail="Chat completion empty.")

    tokens = chat_completion.choices[0].message.content.split()
    return " ".join(tokens[: settings.OUTPUT_MAX_TOKENS]).strip('"')


async def stream_generator(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    tokens_count = 0

    async for chunk in response:
        if not chunk.choices:
            break

        content = chunk.choices[0].delta.content
        tokens_count += len(content.split())

        if settings.OUTPUT_MAX_TOKENS <= tokens_count:
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
        max_tokens=chat_input.max_tokens,
        temperature=settings.llm.TEMPERATURE,
        stream=True,
    )

    return StreamingResponse(stream_generator(response), media_type="text/event-stream")


app.include_router(api_router)


if "__main__" == __name__:
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8005, reload=True, log_level="debug"
    )
