"""Factory pattern for selecting provider implementations based on configuration."""

from typing import Callable

from app.config import settings
from app.providers.base import ToolDefinition
from app.providers.responses import (
    run_responses_inference_batch,
    run_responses_inference_stream,
    stream_generator_responses,
    run_responses_inference_weather,
    ResponsesToolDefinition,
)
from app.providers.chat import (
    run_chat_inference_batch,
    run_chat_inference_stream,
    stream_generator_chat,
    run_chat_inference_weather,
    ChatToolDefinition,
)


def get_inference_callable() -> Callable:
    """
    Factory function to get the appropriate inference provider based on configuration.

    Returns:
        InferenceProvider: Instance of the correct inference provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return run_responses_inference_batch
    else:
        # All other providers use Chat Completions API
        return run_chat_inference_batch


def get_stream_callable() -> Callable:
    """
    Factory function to get the appropriate stream provider based on configuration.

    Returns:
        StreamProvider: Instance of the correct stream provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return run_responses_inference_stream
    else:
        # All other providers use Chat Completions API
        return run_chat_inference_stream


def get_stream_generator() -> Callable:
    """
    Factory function to get the appropriate stream generator based on configuration.

    Returns:
        StreamProvider: Instance of the correct stream generator implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return stream_generator_responses
    else:
        return stream_generator_chat


def get_weather_callable() -> Callable:
    """
    Factory function to get the appropriate weather provider based on configuration.

    Returns:
        StreamProvider: Instance of the correct weather provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return run_responses_inference_weather
    else:
        # All other providers use Chat Completions API
        return run_chat_inference_weather


def get_tool_definitions() -> ToolDefinition:
    """
    Factory function to get the appropriate tool definitions based on configuration.

    Returns:
        ToolDefinition: Instance of the correct tool definitions implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return ResponsesToolDefinition()
    else:
        # All other providers use Chat Completions API format
        return ChatToolDefinition()
