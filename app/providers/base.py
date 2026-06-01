"""Base protocol definitions for provider implementations.

This module defines structural typing protocols that provider implementations
must conform to. Using Protocol (structural typing) instead of ABC allows
for more flexible composition and better type checking support.
"""

from typing import Protocol


class ToolDefinition(Protocol):
    """Protocol for tool definitions.

    Defines the interface for accessing provider-specific tool schemas.
    """

    @property
    def get_current_weather_from_owm(self) -> dict:
        """Get tool definition for weather function."""
        ...
