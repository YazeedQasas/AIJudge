from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App-wide configuration, loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "AI Judge"
    debug: bool = False

    # Prompt version used by default when a request doesn't override it. Rollback =
    # point this at an older version (e.g. "v1") and restart. See app/prompts/.
    active_prompt_version: str = "v1"

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "google/gemma-4-e4b"
    # Max seconds to wait on an LM Studio call. Generous because reasoning models
    # (e.g. gemma) emit many "thinking" tokens first — a full non-streaming answer
    # can take well over a minute. Too low here surfaces in the browser as a
    # misleading CORS error (the 500 timeout response carries no CORS headers).
    llm_timeout_seconds: float = 300.0
    lm_studio_embedding_model: str = "text-embedding-bge-m3"
    embedding_dimension: int = 1024

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_corpus"

    # Heuristic cutoff, not empirically calibrated for this model/corpus. Below this top
    # retrieval score, we refuse to answer rather than hand the LLM weak/irrelevant context.
    min_relevance_score: float = 0.5


settings = Settings()
