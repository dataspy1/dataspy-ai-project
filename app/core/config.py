from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------------
    # App Config
    # -------------------------------
    app_name: str = "DataSpy Decision AI"
    app_env: str = "development"
    debug: bool = True

    # -------------------------------
    # LLM Provider Selection
    # -------------------------------
    llm_provider: str = "groq"  # allowed: "openai", "groq"
    llm_allow_fallback: bool = True
    llm_retry_attempts: int = 2
    llm_retry_delay_seconds: float = 1.0

    # -------------------------------
    # OpenAI Config (fallback / premium)
    # -------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    # -------------------------------
    # Groq Config (fast + cheap)
    # -------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # -------------------------------
    # Future RAG Config (placeholder)
    # -------------------------------
    embedding_provider: str = "huggingface"
    vector_store: str = "faiss"

    # -------------------------------
    # Environment file config
    # -------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def normalized_llm_provider(self) -> str:
        return (self.llm_provider or "").strip().lower()

    @property
    def has_openai_key(self) -> bool:
        return bool((self.openai_api_key or "").strip())

    @property
    def has_groq_key(self) -> bool:
        return bool((self.groq_api_key or "").strip())

    @property
    def validated_llm_provider(self) -> str:
        provider = self.normalized_llm_provider
        allowed = {"groq", "openai"}

        if provider not in allowed:
            raise ValueError(
                f"Unsupported llm_provider='{self.llm_provider}'. "
                f"Allowed values are: {', '.join(sorted(allowed))}."
            )

        return provider


@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    # Early validation so runtime errors are cleaner
    _ = settings.validated_llm_provider

    return settings