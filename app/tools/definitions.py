"""
Tool definitions for LLM functions.

This module provides provider-agnostic access to tool definitions.
The correct format (Responses API vs Chat Completions) is automatically
selected based on DEFAULT_PROVIDER in settings.

Example:
    from app.tools.definitions import GET_CURRENT_WEATHER_FROM_OWM

    tools = [GET_CURRENT_WEATHER_FROM_OWM]
"""

from app.providers.factory import get_tool_definition

# Automatically select the correct tool definition format for the current provider
_tool_def = get_tool_definition()

# Tool schemas
GET_CURRENT_WEATHER_FROM_OWM = _tool_def.get_current_weather_from_owm
CALCULATE = _tool_def.get_calculate

__all__ = ["GET_CURRENT_WEATHER_FROM_OWM", "CALCULATE"]
