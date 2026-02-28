from types import SimpleNamespace

from support_analytics.application.config import AppSettings
from support_analytics.domain.enums import SupportIntent, SupportSatisfaction
from support_analytics.domain.models import ConversationEvaluation
from support_analytics.infrastructure.gemini_client import GoogleGenAIStructuredClient


class UnsupportedStructuredOutputError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_google_genai_client_falls_back_when_native_schema_is_unsupported(monkeypatch) -> None:
    client = object.__new__(GoogleGenAIStructuredClient)
    client._settings = AppSettings(google_api_key="test", _env_file=None)
    client._client = SimpleNamespace()

    def fake_generate_content_with_retry(**_: object) -> object:
        raise UnsupportedStructuredOutputError(400, "JSON mode is not enabled for this model")

    def fake_generate_plain_content_with_retry(**_: object) -> object:
        return SimpleNamespace(
            text='{"intent":"technical_error","satisfaction":"unsatisfied","quality_score":2,"agent_mistakes":["no_resolution"]}'
        )

    monkeypatch.setattr(
        "support_analytics.infrastructure.gemini_client.ClientError",
        UnsupportedStructuredOutputError,
    )
    client._generate_content_with_retry = fake_generate_content_with_retry  # type: ignore[method-assign]
    client._generate_plain_content_with_retry = fake_generate_plain_content_with_retry  # type: ignore[method-assign]

    evaluation = client.generate_structured(
        model="gemma-3-27b-it",
        prompt="test",
        system_instruction="system",
        response_schema=ConversationEvaluation,
        seed=42,
    )

    assert evaluation.intent == SupportIntent.TECHNICAL_ERROR
    assert evaluation.satisfaction == SupportSatisfaction.UNSATISFIED
    assert evaluation.quality_score == 2
    assert len(evaluation.agent_mistakes) == 1
