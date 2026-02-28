from support_analytics.application.config import AppSettings
from support_analytics.infrastructure.gemini_client import GoogleGenAIStructuredClient


class DummyClientError(Exception):
    def __init__(self, code: int, message: str, details: object | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def test_google_genai_retry_delay_prefers_server_retry_hint() -> None:
    client = object.__new__(GoogleGenAIStructuredClient)
    client._settings = AppSettings(
        google_api_key="test",
        llm_retry_base_delay_seconds=1.0,
        llm_retry_max_delay_seconds=60.0,
        _env_file=None,
    )
    error = DummyClientError(
        429,
        "rate limited",
        details={"error": {"details": [{"retryDelay": "8s"}]}},
    )

    delay = client._retry_delay_seconds(0, error=error)  # type: ignore[arg-type]

    assert delay == 8.0


def test_google_genai_detects_non_transient_quota_exhaustion() -> None:
    client = object.__new__(GoogleGenAIStructuredClient)
    client._settings = AppSettings(
        google_api_key="test",
        llm_fail_fast_on_quota_exhaustion=True,
        _env_file=None,
    )
    error = DummyClientError(
        429,
        "Quota exceeded for limit 'GenerateContentRequestsPerDayPerProjectPerModel-FreeTier'.",
    )

    should_retry = client._should_retry(error=error, attempt=0)  # type: ignore[arg-type]

    assert should_retry is False


def test_google_genai_default_pacing_for_flash_lite_is_conservative() -> None:
    client = object.__new__(GoogleGenAIStructuredClient)
    client._settings = AppSettings(google_api_key="test", _env_file=None)

    delay = client._minimum_request_interval_seconds_for_model("gemini-2.5-flash-lite")

    assert delay >= 4.0
