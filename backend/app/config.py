from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App-wide configuration, loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "AI Judge"
    debug: bool = False

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "google/gemma-4-e4b"
    lm_studio_embedding_model: str = "text-embedding-bge-m3"
    embedding_dimension: int = 1024

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_corpus"

    # Heuristic cutoff, not empirically calibrated for this model/corpus. Below this top
    # retrieval score, we refuse to answer rather than hand the LLM weak/irrelevant context.
    min_relevance_score: float = 0.5


settings = Settings()
