"""
This file contains json-schemas for the tools

"""

GET_CURRENT_WEATHER_FROM_OWM = {
    "type": "function",
    "function": {
        "name": "get_current_weather_from_owm",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or state, e.g. San Francisco, CA",
                },
                "unit_sys": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["location"],
        },
    },
}
