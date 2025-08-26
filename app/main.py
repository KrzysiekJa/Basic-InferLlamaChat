from fastapi import APIRouter, Depends, FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from app import deps
from app.config import settings


app = FastAPI(title="Llama4Infer ChatApp")
api_router = APIRouter()


@api_router.get("/", status_code=200)
async def root():
    return "Navigate to /docs to see the API documentation."


class ChatInput(BaseModel):
    user_prompt: str = "Tell me about Nicolas Cage."
    max_tokens: int = 128


@api_router.post("/inference/batch", status_code=200)
async def run_chat_inference_batch(
    chat_input: ChatInput, llm_client: AsyncOpenAI | None = Depends(deps.get_llm_client)
):
    chat_completion = await llm_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": chat_input.user_prompt[: chat_input.max_tokens],
            },
        ],
        model=settings.llm.MODEL,
        max_completion_tokens=settings.llm.MAX_TOKENS,
        temperature=settings.llm.TEMPERATURE,
    )

    if not chat_completion:
        raise HTTPException(status_code=404, detail="Chat completion empty.")

    return chat_completion.choices[0].message.content.strip()


app.include_router(api_router)


if "__main__" == __name__:
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8005, reload=True, log_level="debug"
    )
