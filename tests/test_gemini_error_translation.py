from support_analytics.application.config import AppSettings
from support_analytics.infrastructure.gemini_client import GeminiStructuredClient


class DummyClientError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_leaked_key_error_is_translated() -> None:
    client = object.__new__(GeminiStructuredClient)
    client._settings = AppSettings(google_api_key="test", _env_file=None)
    error = DummyClientError(403, "Your API key was reported as leaked. Please use another API key.")

    translated = client._translate_client_error(error)  # type: ignore[arg-type]

    assert "flagged it as leaked" in str(translated)
    assert "Create a new key" in str(translated)
