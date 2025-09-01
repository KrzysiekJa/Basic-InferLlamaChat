import json
import requests

from app.config import settings


def get_current_weather_from_owm(location: str, unit_sys: str = "metric") -> str:
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
