from pydantic import BaseModel

from app.config import settings


class ChatInput(BaseModel):
    user_prompt: str = "Tell me about Nicolas Cage."
    max_tokens: int = settings.llm.MAX_TOKENS


class WeatherInput(BaseModel):
    user_prompt: str = "Bergamo, Italy"
    max_tokens: int = settings.weather_api.MAX_TOKENS
