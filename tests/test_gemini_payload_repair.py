from support_analytics.domain.models import ConversationDraft
from support_analytics.infrastructure.gemini_client import (
    build_response_json_schema,
    repair_payload_to_schema_constraints,
)


def test_repair_payload_truncates_string_fields_to_schema_limits() -> None:
    payload = {
        "title": "T" * 140,
        "scenario_summary": "S" * 400,
        "turns": [
            {"role": "customer", "message": "A" * 1200},
            {"role": "agent", "message": "B" * 1200},
            {"role": "customer", "message": "C" * 1200},
            {"role": "agent", "message": "D" * 1200},
        ],
        "final_customer_signal": "F" * 240,
    }

    repaired = repair_payload_to_schema_constraints(
        payload,
        build_response_json_schema(ConversationDraft),
    )
    draft = ConversationDraft.model_validate(repaired)

    assert len(draft.title) <= 120
    assert len(draft.scenario_summary) <= 320
    assert len(draft.final_customer_signal) <= 200
    assert all(len(turn.message) <= 1000 for turn in draft.turns)
