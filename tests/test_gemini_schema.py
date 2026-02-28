import json

from support_analytics.domain.models import ConversationDraft
from support_analytics.infrastructure.gemini_client import build_response_json_schema


def test_build_response_json_schema_removes_problematic_fields() -> None:
    schema = build_response_json_schema(ConversationDraft)
    schema_json = json.dumps(schema, ensure_ascii=False)

    assert "$defs" not in schema
    assert "$ref" not in schema_json
    assert "additionalProperties" not in schema_json
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "title" in schema["properties"]
    assert "turns" in schema["properties"]
