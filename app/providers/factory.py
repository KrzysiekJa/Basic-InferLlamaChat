"""Factory pattern for selecting provider implementations based on configuration."""

from app.config import settings
from app.providers.base import InferenceProvider, StreamProvider, ToolDefinition
from app.providers.responses import (
    ResponsesInferenceProvider,
    ResponsesStreamProvider,
    ResponsesToolDefinition,
)
from app.providers.chat import ChatInferenceProvider, ChatStreamProvider, ChatToolDefinition


def get_inference_provider() -> InferenceProvider:
    """
    Factory function to get the appropriate inference provider based on configuration.
    
    Returns:
        InferenceProvider: Instance of the correct inference provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return ResponsesInferenceProvider()
    else:
        # All other providers use Chat Completions API
        return ChatInferenceProvider()


def get_stream_provider() -> StreamProvider:
    """
    Factory function to get the appropriate stream provider based on configuration.
    
    Returns:
        StreamProvider: Instance of the correct stream provider implementation.
    """
    if settings.llm.DEFAULT_PROVIDER == "openai":
        return ResponsesStreamProvider()
    else:
        # All other providers use Chat Completions API
        return ChatStreamProvider()


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
