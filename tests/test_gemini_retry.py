from support_analytics.application.config import AppSettings
from support_analytics.infrastructure.gemini_client import GeminiStructuredClient


class FakeRateLimitError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, failures_before_success: int, response: object) -> None:
        self.failures_before_success = failures_before_success
        self.response = response
        self.calls = 0

    def generate_content(self, **_: object) -> object:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise FakeRateLimitError(429, "rate limited")
        return self.response


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def test_generate_content_retries_after_rate_limit(monkeypatch) -> None:
    client = object.__new__(GeminiStructuredClient)
    client._settings = AppSettings(
        google_api_key="test",
        llm_max_retries=3,
        llm_retry_base_delay_seconds=0.01,
        llm_retry_max_delay_seconds=0.01,
        _env_file=None,
    )
    fake_models = FakeModels(failures_before_success=2, response={"ok": True})
    client._client = FakeClient(fake_models)

    sleep_calls: list[float] = []
    monkeypatch.setattr("support_analytics.infrastructure.gemini_client.ClientError", FakeRateLimitError)
    monkeypatch.setattr("support_analytics.infrastructure.gemini_client.time.sleep", sleep_calls.append)

    response = client._generate_content_with_retry(
        model="gemini-2.5-flash-lite",
        prompt="test",
        system_instruction="system",
        json_schema={"type": "object"},
        seed=42,
    )

    assert response == {"ok": True}
    assert fake_models.calls == 3
    assert sleep_calls == [0.01, 0.01]
