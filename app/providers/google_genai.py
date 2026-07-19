"""Google GenAI provider implementation using google-genai SDK.

This module provides the same provider functions as other providers so the
factory can select it transparently.
"""

from typing import AsyncGenerator
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from google.genai import types

from app.config import settings


async def run_google_inference_batch(
    user_prompt: str, max_tokens: int, llm_client
) -> str:
    # llm_client expected to be an async google-genai client (client.aio)
    response = await llm_client.models.generate_content(
        model=settings.llm.GOOGLE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=settings.llm.TEMPERATURE,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        ),
    )

    if not response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response is empty."
        )

    return response.text


async def run_google_inference_stream(
    user_prompt: str, max_tokens: int, llm_client
) -> StreamingResponse:
    response = await llm_client.models.generate_content_stream(
        model=settings.llm.GOOGLE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=settings.llm.TEMPERATURE,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        ),
    )

    return StreamingResponse(
        stream_generator_google(response), media_type="text/event-stream"
    )


async def run_google_inference_weather(
    user_prompt: str, max_tokens: int, llm_client
) -> str:
    from app.providers.factory import get_tool_definition
    from app.tools.functions import get_current_weather_from_owm

    tool_defs = get_tool_definition()
    tool = tool_defs.get_current_weather_from_owm

    # Disable automatic function calling so we handle the function invocation manually
    response = await llm_client.models.generate_content(
        model=settings.llm.GOOGLE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=settings.weather_api.MAX_TOKENS,
            tools=[tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        ),
    )

    function_calls = getattr(response, "function_calls", None) or []

    if not function_calls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response output is empty."
        )

    function_call = function_calls[0]
    function_name = function_call.name
    function_args = getattr(function_call, "args", {})

    if function_name == "get_current_weather_from_owm":
        function_response = get_current_weather_from_owm(
            function_args.get("location"), function_args.get("unit", "metric")
        )

        # Build a function response part and pass it back to the model
        function_response_part = types.Part.from_function_response(
            name=function_name, response={"result": function_response}
        )

        function_response_content = types.Content(
            role="tool", parts=[function_response_part]
        )

        enriched = await llm_client.models.generate_content(
            model=settings.llm.GOOGLE_MODEL,
            contents=[user_prompt, function_response_content],
            config=types.GenerateContentConfig(tools=[tool]),
        )

        if not enriched or not getattr(enriched, "text", None):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enriched response is empty.",
            )

        return enriched.text

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported function call."
    )


async def run_google_inference_calculator(
    user_prompt: str, max_tokens: int, llm_client
) -> str:
    from app.providers.factory import get_tool_definition
    from app.tools.functions import calculate

    tool_defs = get_tool_definition()
    tool = tool_defs.get_calculate

    # Disable automatic function calling so we handle the function invocation manually
    response = await llm_client.models.generate_content(
        model=settings.llm.GOOGLE_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=settings.weather_api.MAX_TOKENS,
            tools=[tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
        ),
    )

    function_calls = getattr(response, "function_calls", None) or []

    if not function_calls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Response output is empty."
        )

    function_call = function_calls[0]
    function_name = function_call.name
    function_args = getattr(function_call, "args", {})

    if function_name == "calculate":
        function_response = calculate(
            function_args.get("operation"),
            float(function_args.get("x", 0)),
            float(function_args.get("y", 0)),
        )

        # Build a function response part and pass it back to the model
        function_response_part = types.Part.from_function_response(
            name=function_name, response={"result": function_response}
        )

        function_response_content = types.Content(
            role="tool", parts=[function_response_part]
        )

        enriched = await llm_client.models.generate_content(
            model=settings.llm.GOOGLE_MODEL,
            contents=[user_prompt, function_response_content],
            config=types.GenerateContentConfig(tools=[tool]),
        )

        if not enriched or not getattr(enriched, "text", None):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Enriched response is empty.",
            )

        return enriched.text

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported function call."
    )


async def stream_generator_google(
    response: AsyncGenerator,
) -> AsyncGenerator[str, None]:
    tokens_count = 0

    async for chunk in response:
        text = getattr(chunk, "text", None)
        if not text:
            break

        tokens_count += len(text.split())
        if settings.chat.OUTPUT_MAX_TOKENS <= tokens_count:
            break

        yield text


class GoogleGenAIToolDefinition:
    """Tool definitions adapted to google-genai types."""

    @property
    def get_current_weather_from_owm(self):
        function = types.FunctionDeclaration(
            name="get_current_weather_from_owm",
            description="Get the current weather in a given location",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or state"},
                    "unit": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["location"],
            },
        )

        return types.Tool(function_declarations=[function])

    @property
    def get_calculate(self):
        function = types.FunctionDeclaration(
            name="calculate",
            description="Perform a basic arithmetic operation on two numbers.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform.",
                    },
                    "x": {"type": "number", "description": "The first operand."},
                    "y": {"type": "number", "description": "The second operand."},
                },
                "required": ["operation", "x", "y"],
            },
        )

        return types.Tool(function_declarations=[function])
