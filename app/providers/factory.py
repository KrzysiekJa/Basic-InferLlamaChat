"""Factory pattern for selecting provider implementations based on configuration."""

from typing import Callable

from app.config import settings
from app.providers.base import ToolDefinition
from app.providers.responses import (
    run_responses_inference_batch,
    run_responses_inference_stream,
    stream_generator_responses,
    run_responses_inference_weather,
    run_responses_inference_calculator,
    ResponsesToolDefinition,
)
from app.providers.chat import (
    run_chat_inference_batch,
    run_chat_inference_stream,
    stream_generator_chat,
    run_chat_inference_weather,
    run_chat_inference_calculator,
    ChatToolDefinition,
)
from app.providers import google_genai


def get_inference_callable() -> Callable:
    """
    Factory function to get the appropriate inference provider based on configuration.

    Returns:
        InferenceProvider: Instance of the correct inference provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return run_responses_inference_batch
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.run_google_inference_batch
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
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.run_google_inference_stream
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
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.stream_generator_google
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
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.run_google_inference_weather
    else:
        # All other providers use Chat Completions API
        return run_chat_inference_weather


def get_calculator_callable() -> Callable:
    """
    Factory function to get the appropriate calculator provider based on configuration.

    Returns:
        Callable: The correct calculator provider implementation for the active provider.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return run_responses_inference_calculator
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.run_google_inference_calculator
    else:
        # All other providers use Chat Completions API
        return run_chat_inference_calculator


def get_tool_definition() -> ToolDefinition:
    """
    Factory function to get the appropriate tool definitions based on configuration.

    Returns:
        ToolDefinition: Instance of the correct tool definitions implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return ResponsesToolDefinition()
    if settings.llm.DEFAULT_PROVIDER == "google":
        return google_genai.GoogleGenAIToolDefinition()
    else:
        # All other providers use Chat Completions API format
        return ChatToolDefinition()
