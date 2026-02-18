from pydantic import BaseModel, Field

from app.config import settings


class ChatInput(BaseModel):
    user_prompt: str = Field("Tell me about Nicolas Cage.")
    max_tokens: int = Field(settings.llm.MAX_TOKENS)


class WeatherInput(BaseModel):
    user_prompt: str = Field("Bergamo, Italy")
    max_tokens: int = Field(settings.weather_api.MAX_TOKENS)
