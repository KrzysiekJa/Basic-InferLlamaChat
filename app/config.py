import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


BASE_PATH = Path(__file__).resolve().parent


class LLMSettings(BaseSettings):
    # will be read from .env variables
    CONTEXT_WINDOW: int
    MAX_TOKENS: int
    TEMPERATURE: float
    MODEL: str
    DEFAULT_PROVIDER: str
    TOGETHER_API_KEY: str
    TOGETHER_API_URL: str
    OPENROUTER_API_KEY: str
    OPENROUTER_API_URL: str
    OPENAI_API_KEY: str
    OPENAI_API_URL: str
    GOOGLE_API_KEY: str
    GOOGLE_MODEL: str
    API_KEY: str = ""
    BASE_URL: str = ""

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"

    @model_validator(mode="after")
    def set_api_key(self) -> "LLMSettings":
        # Set API_KEY and BASE_URL based on provider

        match self.DEFAULT_PROVIDER:
            case "together":
                self.API_KEY = self.TOGETHER_API_KEY
                self.BASE_URL = self.TOGETHER_API_URL
            case "openrouter":
                self.API_KEY = self.OPENROUTER_API_KEY
                self.BASE_URL = self.OPENROUTER_API_URL
            case "openai":
                self.API_KEY = self.OPENAI_API_KEY
                self.BASE_URL = self.OPENAI_API_URL
            case "google":
                self.API_KEY = self.GOOGLE_API_KEY
                # google-genai client manages endpoints internally
                self.BASE_URL = ""
            case _:
                self.API_KEY = self.OPENAI_API_KEY
                self.BASE_URL = self.OPENAI_API_URL

        return self


class WeatherAPISettings(BaseSettings):
    OWM_API_KEY: str
    BASE_URL: str
    MAX_TOKENS: int = os.getenv("OWM_MAX_TOKENS", 128)

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"


class ChatSettings(BaseSettings):
    OUTPUT_MIN_TOKENS: int
    OUTPUT_MAX_TOKENS: int

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    weather_api: WeatherAPISettings = WeatherAPISettings()
    chat: ChatSettings = ChatSettings()

    class Config:
        case_sensitive = True


settings = Settings()
