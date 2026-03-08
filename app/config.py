from pathlib import Path

from pydantic_settings import BaseSettings


BASE_PATH = Path(__file__).resolve().parent


class LLMSettings(BaseSettings):
    CONTEXT_WINDOW: int = 16000
    MAX_TOKENS: int = 128
    TEMPERATURE: float = 0.7
    MODEL: str = "openai/gpt-5-nano" # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    OPENAI_API_KEY: str  # will be read from env variable
    BASE_URL: str = "https://api.openai.com"


class WeatherAPISettings(BaseSettings):
    OWM_API_KEY: str  # openweathermap api key
    BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather?"
    MAX_TOKENS: int = 128

class ChatSettings(BaseSettings):
    OUTPUT_MIN_TOKENS: int = 0
    OUTPUT_MAX_TOKENS: int = 768


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    weather_api: WeatherAPISettings = WeatherAPISettings()
    chat: ChatSettings = ChatSettings()

    class Config:
        case_sensitive = True


settings = Settings()
