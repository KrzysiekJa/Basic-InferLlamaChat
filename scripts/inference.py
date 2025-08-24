import os
import json
from typing import Any, Generator
from openai import OpenAI
from pydantic import BaseModel, Field

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", None)
client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz")


def run_batch_inference(
    user_prompt: str, system_message: str = "You are a helpful AI assistant."
) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
    )
    return chat_completion.choices[0].message.content


def run_stream_inference(
    user_prompt: str, system_message: str = "You are a helpful AI assistant."
) -> Generator[Any, Any, Any]:
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
        stream=True,
    )

    for chunk in stream:
        try:
            chunk_content = chunk.choices[0].delta.content or ""
            yield chunk_content
        except IndexError:
            # Handle the case where chunk.choices is empty
            yield ""


class User(BaseModel):
    name: str = Field(..., description="The name of the user", example="John Doe")
    address: str = Field(
        ..., description="The address of the user", example="123 Main St"
    )


def get_inference_schema(
    user_prompt: str = "Create a user named John Doe, who lives at 123 Main St.",
    system_message: str = "You are a helpful AI assistant.",
) -> str:
    chat_completion = client.chat.completions.create(
        response_format={"type": "json_object", "schema": User.model_json_schema()},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
    )

    create_user = json.loads(chat_completion.choices[0].message.content)
    return create_user


def main():
    result = run_batch_inference("Tell about Paris.")
    print(result)

    for part in run_stream_inference("Tell me about Madrid and Warsaw."):
        print(part, end="", flush=True)

    print("\n")

    result = get_inference_schema(
        user_prompt="Create a user named Jane Doe, who lives at 97, Smith Street."
    )
    print(result)


if __name__ == "__main__":
    main()
