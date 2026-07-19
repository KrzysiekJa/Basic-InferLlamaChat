import json
import requests

from app.config import settings


def calculate(operation: str, x: float, y: float) -> str:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        operation (str): One of "add", "subtract", "multiply", "divide".
        x (float): The first operand.
        y (float): The second operand.

    Returns:
        str: A JSON string containing the operation, operands, and result.
             For division by zero, returns a JSON error string.
    """
    operation = (operation or "").lower()

    if operation == "add":
        result = x + y
    elif operation == "subtract":
        result = x - y
    elif operation == "multiply":
        result = x * y
    elif operation == "divide":
        if y == 0:
            return json.dumps(
                {
                    "operation": operation,
                    "x": x,
                    "y": y,
                    "error": "division by zero",
                }
            )
        result = x / y
    else:
        return json.dumps(
            {
                "operation": operation,
                "x": x,
                "y": y,
                "error": f"unsupported operation: {operation}",
            }
        )

    return json.dumps(
        {
            "operation": operation,
            "x": x,
            "y": y,
            "result": result,
        }
    )


def get_current_weather_from_owm(location: str, unit_sys: str = "metric") -> str:
    """Get current weather information for a given location.

    Args:
        location (str): The location for which to retrieve weather information.
        unit_sys (str, optional): The unit system for the weather data. Defaults to "metric".

    Returns:
        str: A JSON string containing the weather information.
    """
    base_url, api_key = settings.weather_api.BASE_URL, settings.weather_api.OWM_API_KEY
    url = f"{base_url}q={location}&appid={api_key}&units={unit_sys}"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return json.dumps(
            {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "pressure_unit": "hPa" if unit_sys == "metric" else "inHg",
                "feels_like": data["main"]["feels_like"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"],
                "temperature_unit": "°C"
                if unit_sys == "metric"
                else "°F"
                if unit_sys == "imperial"
                else "K",
                "speed_unit": "m/s" if unit_sys == "metric" else "mph",
            }
        )
    return json.dumps(
        {
            "location": location,
            "temperature": "unknown",
            "description": "unknown",
        }
    )
