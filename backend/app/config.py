from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://docrag:docrag@localhost:5432/docrag"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "./uploads"

    max_upload_mb: int = 10
    embed_dim: int = 256
    chunk_words: int = 180
    chunk_overlap: int = 40
    top_k: int = 4

    # LLM is optional. Without a key the service answers extractively.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"
    llm_max_tokens: int = 400

    # Denial-of-wallet / abuse protection (OWASP LLM10).
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 20
    max_question_chars: int = 500

    cors_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
