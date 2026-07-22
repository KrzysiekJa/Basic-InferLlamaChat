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
    api_key: str = ""
    base_url: str = ""

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"

    @model_validator(mode="after")
    def set_api_key(self) -> "LLMSettings":
        # Set API_KEY and BASE_URL based on provider

        match self.DEFAULT_PROVIDER:
            case "together":
                self.api_key = self.TOGETHER_API_KEY
                self.base_url = self.TOGETHER_API_URL
            case "openrouter":
                self.api_key = self.OPENROUTER_API_KEY
                self.base_url = self.OPENROUTER_API_URL
            case "openai":
                self.api_key = self.OPENAI_API_KEY
                self.base_url = self.OPENAI_API_URL
            case "google":
                self.api_key = self.GOOGLE_API_KEY
                # google-genai client manages endpoints internally
                self.base_url = ""
            case _:
                self.api_key = self.OPENAI_API_KEY
                self.base_url = self.OPENAI_API_URL

        return self


class WeatherAPISettings(BaseSettings):
    OWM_API_KEY: str
    BASE_URL: str
    MAX_TOKENS: int = int(os.getenv("OWM_MAX_TOKENS", "128"))

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"


class CalculatorSettings(BaseSettings):
    MAX_TOKENS: int

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


class RestSettings(BaseSettings):
    ORIGINS: list[str]
    ALLOWED_CREDENTIALS: bool
    METHODS: list[str]
    HEADERS: list[str]

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"


class UvicornSettings(BaseSettings):
    APP_PATH: str
    IP: str
    PORT: int
    RELOAD: bool
    LOG_LEVEL: str

    class Config:
        env_file = "app/.env"
        case_sensitive = True
        extra = "ignore"


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    weather_api: WeatherAPISettings = WeatherAPISettings()
    calculator: CalculatorSettings = CalculatorSettings()
    chat: ChatSettings = ChatSettings()
    rest: RestSettings = RestSettings()
    uvicorn: UvicornSettings = UvicornSettings()

    class Config:
        case_sensitive = True


settings = Settings()
