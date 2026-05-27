"""Base protocol definitions for provider implementations.

This module defines structural typing protocols that provider implementations
must conform to. Using Protocol (structural typing) instead of ABC allows
for more flexible composition and better type checking support.
"""

from typing import Protocol, AsyncGenerator, Any
from openai import AsyncOpenAI


class InferenceProvider(Protocol):
    """Protocol for inference providers.
    
    Any class implementing these methods conforms to the InferenceProvider protocol,
    even without explicitly inheriting from it.
    """

    async def inference_batch(
        self, user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
    ) -> str:
        """Perform batch inference."""
        ...

    async def inference_stream(
        self, user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
    ) -> Any:
        """Perform streaming inference."""
        ...

    async def inference_weather(
        self, user_prompt: str, max_tokens: int, llm_client: AsyncOpenAI
    ) -> str:
        """Perform inference with weather tool."""
        ...


class StreamProvider(Protocol):
    """Protocol for stream generators.
    
    Defines the interface for generating streaming output from various providers.
    """

    async def stream_generator(self, response: AsyncGenerator) -> AsyncGenerator[str, None]:
        """Generate streaming output."""
        ...


class ToolDefinition(Protocol):
    """Protocol for tool definitions.
    
    Defines the interface for accessing provider-specific tool schemas.
    """

    @property
    def get_current_weather_from_owm(self) -> dict:
        """Get tool definition for weather function."""
        ...

