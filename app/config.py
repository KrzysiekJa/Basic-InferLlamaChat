from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    CONTEXT_WINDOW: int = 16000
    MAX_TOKENS: int = 768
    TEMPERATURE: float = 0.7
    MODEL: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    TOGETHER_API_KEY: str  # will be read from env variable
    BASE_URL: str = "https://api.together.xyz"


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()

    class Config:
        case_sensitive = True


settings = Settings()
