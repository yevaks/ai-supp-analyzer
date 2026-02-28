from support_analytics.application.config import AppSettings
from support_analytics.infrastructure import llm_factory


def test_build_structured_llm_uses_google_genai_provider(monkeypatch) -> None:
    created: list[str] = []

    class FakeGoogleClient:
        def __init__(self, settings: AppSettings) -> None:
            created.append(settings.llm_provider)

    monkeypatch.setattr(llm_factory, "GoogleGenAIStructuredClient", FakeGoogleClient)

    llm_factory.build_structured_llm(
        AppSettings(
            llm_provider="google_genai",
            google_api_key="test",
            _env_file=None,
        )
    )

    assert created == ["google_genai"]


def test_build_structured_llm_rejects_unknown_provider() -> None:
    settings = AppSettings(
        llm_provider="unsupported",
        google_api_key="test",
        _env_file=None,
    )

    try:
        llm_factory.build_structured_llm(settings)
    except RuntimeError as error:
        assert "Unsupported LLM_PROVIDER" in str(error)
    else:
        raise AssertionError("Expected RuntimeError for unsupported provider.")
