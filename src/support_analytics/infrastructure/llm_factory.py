from __future__ import annotations

from support_analytics.application.config import AppSettings
from support_analytics.application.ports import StructuredLLM
from support_analytics.infrastructure.gemini_client import GoogleGenAIStructuredClient


def build_structured_llm(settings: AppSettings) -> StructuredLLM:
    if settings.llm_provider in {"google_genai", "google", "gemini_api"}:
        return GoogleGenAIStructuredClient(settings)
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
        "This build currently supports Google GenAI hosted models such as Gemini and Gemma."
    )
