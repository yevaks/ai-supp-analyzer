from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: str = Field(default="google_genai", validation_alias="LLM_PROVIDER")
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "LLM_API_KEY", "GEMINI_API_KEY"),
    )
    llm_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MODEL", "GEMINI_MODEL"),
    )
    generation_model: str | None = Field(default=None, validation_alias="GENERATION_MODEL")
    evaluation_model: str | None = Field(default=None, validation_alias="EVALUATION_MODEL")
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        validation_alias="GEMINI_MODEL",
    )
    dataset_language: str = Field(default="uk", validation_alias="DATASET_LANGUAGE")
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    candidate_count: int = 1
    max_output_tokens: int = 4096
    llm_max_retries: int = Field(
        default=10,
        validation_alias=AliasChoices("LLM_MAX_RETRIES", "GEMINI_MAX_RETRIES"),
    )
    llm_retry_base_delay_seconds: float = Field(
        default=3.0,
        validation_alias=AliasChoices("LLM_RETRY_BASE_DELAY_SECONDS", "GEMINI_RETRY_BASE_DELAY_SECONDS"),
    )
    llm_retry_max_delay_seconds: float = Field(
        default=60.0,
        validation_alias=AliasChoices("LLM_RETRY_MAX_DELAY_SECONDS", "GEMINI_RETRY_MAX_DELAY_SECONDS"),
    )
    cache_dir: Path = Path(".cache/support-analytics")
    artifacts_dir: Path = Path("artifacts")

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "Google API key is not set. Add `GOOGLE_API_KEY`, `LLM_API_KEY`, or `GEMINI_API_KEY` "
                "to the environment or .env file."
            )
        return self.google_api_key

    def resolve_generation_model(self, override: str | None = None) -> str:
        if override:
            return override
        if self.generation_model:
            return self.generation_model
        if self.llm_model:
            return self.llm_model
        return self.gemini_model

    def resolve_evaluation_model(self, override: str | None = None) -> str:
        if override:
            return override
        if self.evaluation_model:
            return self.evaluation_model
        if self.llm_model:
            return self.llm_model
        return self.gemini_model
