from support_analytics.application.config import AppSettings


def test_resolve_models_prefers_explicit_generation_and_evaluation_models() -> None:
    settings = AppSettings(
        google_api_key="test",
        llm_model="gemma-3-27b-it",
        generation_model="gemini-2.5-flash-lite",
        evaluation_model="gemma-3-27b-it",
        _env_file=None,
    )

    assert settings.resolve_generation_model() == "gemini-2.5-flash-lite"
    assert settings.resolve_evaluation_model() == "gemma-3-27b-it"


def test_resolve_models_falls_back_to_shared_llm_model() -> None:
    settings = AppSettings(
        google_api_key="test",
        llm_model="gemma-3-27b-it",
        _env_file=None,
    )

    assert settings.resolve_generation_model() == "gemma-3-27b-it"
    assert settings.resolve_evaluation_model() == "gemma-3-27b-it"
