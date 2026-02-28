from __future__ import annotations

import json
import re
import time
from typing import Any, TypeVar

from google import genai
from google.genai.errors import ClientError
from google.genai import types
from pydantic import BaseModel, ValidationError

from support_analytics.application.config import AppSettings
from support_analytics.application.ports import StructuredLLM

T = TypeVar("T", bound=BaseModel)


def build_response_json_schema(response_schema: type[BaseModel]) -> dict[str, Any]:
    """Build a Gemini-compatible JSON schema from a Pydantic model.

    Gemini's `response_schema` field uses a reduced Schema object and rejects
    `additional_properties`. Passing native JSON Schema via
    `response_json_schema` is more reliable, but we still strip unsupported or
    unnecessary fields and inline refs for predictable requests.
    """

    raw_schema = response_schema.model_json_schema()
    definitions = raw_schema.get("$defs", {})

    def _normalize(node: Any, *, preserve_mapping_keys: bool = False) -> Any:
        if isinstance(node, list):
            return [_normalize(item) for item in node]

        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = str(node["$ref"]).split("/")[-1]
            resolved = definitions.get(ref_name, {})
            merged = {**resolved, **{key: value for key, value in node.items() if key != "$ref"}}
            return _normalize(merged)

        if preserve_mapping_keys:
            return {key: _normalize(value) for key, value in node.items()}

        normalized: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"$defs", "$schema", "additionalProperties", "title", "default"}:
                continue
            if key == "properties":
                normalized[key] = _normalize(value, preserve_mapping_keys=True)
                continue
            normalized[key] = _normalize(value)
        return normalized

    return _normalize(raw_schema)


def repair_payload_to_schema_constraints(payload: Any, schema: dict[str, Any]) -> Any:
    """Apply deterministic truncation to common JSON-schema length violations."""

    schema_type = schema.get("type")

    if schema_type == "object" and isinstance(payload, dict):
        properties = schema.get("properties", {})
        repaired: dict[str, Any] = {}
        for key, value in payload.items():
            property_schema = properties.get(key)
            repaired[key] = (
                repair_payload_to_schema_constraints(value, property_schema)
                if isinstance(property_schema, dict)
                else value
            )
        return repaired

    if schema_type == "array" and isinstance(payload, list):
        max_items = schema.get("maxItems")
        repaired_items = payload[:max_items] if isinstance(max_items, int) else list(payload)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [
                repair_payload_to_schema_constraints(item, item_schema)
                for item in repaired_items
            ]
        return repaired_items

    if schema_type == "string" and isinstance(payload, str):
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(payload) > max_length:
            return payload[: max(0, max_length - 1)].rstrip() + "…"
        return payload.strip()

    return payload



class GoogleGenAIStructuredClient(StructuredLLM):
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.require_api_key())
        self._next_request_not_before_by_model: dict[str, float] = {}

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: type[T],
        seed: int,
    ) -> T:
        json_schema = build_response_json_schema(response_schema)
        try:
            response = self._generate_content_with_retry(
                model=model,
                prompt=prompt,
                system_instruction=system_instruction,
                json_schema=json_schema,
                seed=seed,
            )
            payload = response.parsed if response.parsed is not None else json.loads(response.text)
        except ClientError as error:
            if not self._should_fallback_to_prompt_json(error):
                raise self._translate_client_error(error) from error
            payload = self._generate_prompt_json_fallback(
                model=model,
                prompt=prompt,
                system_instruction=system_instruction,
                json_schema=json_schema,
                seed=seed,
            )
        try:
            return response_schema.model_validate(payload)
        except ValidationError:
            repaired_payload = repair_payload_to_schema_constraints(payload, json_schema)
            try:
                return response_schema.model_validate(repaired_payload)
            except ValidationError as error:
                raise RuntimeError(
                    f"Google model returned structured data that failed validation after repair: {error}"
                ) from error

    def _generate_prompt_json_fallback(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        json_schema: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        fallback_prompt = self._build_prompt_json_request(
            prompt=prompt,
            system_instruction=system_instruction,
            json_schema=json_schema,
        )
        try:
            response = self._generate_plain_content_with_retry(
                model=model,
                prompt=fallback_prompt,
                seed=seed,
            )
        except ClientError as error:
            raise self._translate_client_error(error) from error
        return self._extract_json_payload(response.text)

    def _generate_content_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        json_schema: dict[str, Any],
        seed: int,
    ) -> types.GenerateContentResponse:
        attempts = self._settings.llm_max_retries + 1
        last_error: ClientError | None = None

        for attempt in range(attempts):
            try:
                self._wait_for_request_slot(model)
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_json_schema=json_schema,
                        temperature=self._settings.temperature,
                        top_p=self._settings.top_p,
                        top_k=self._settings.top_k,
                        candidate_count=self._settings.candidate_count,
                        max_output_tokens=self._settings.max_output_tokens,
                        seed=seed,
                    ),
                )
            except ClientError as error:
                last_error = error
                if not self._should_retry(error=error, attempt=attempt):
                    raise
                time.sleep(self._retry_delay_seconds(attempt, error=error))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without a response or error.")

    def _generate_plain_content_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        seed: int,
    ) -> types.GenerateContentResponse:
        attempts = self._settings.llm_max_retries + 1
        last_error: ClientError | None = None

        for attempt in range(attempts):
            try:
                self._wait_for_request_slot(model)
                return self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self._settings.temperature,
                        top_p=self._settings.top_p,
                        top_k=self._settings.top_k,
                        candidate_count=self._settings.candidate_count,
                        max_output_tokens=self._settings.max_output_tokens,
                        seed=seed,
                    ),
                )
            except ClientError as error:
                last_error = error
                if not self._should_retry(error=error, attempt=attempt):
                    raise
                time.sleep(self._retry_delay_seconds(attempt, error=error))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini fallback request failed without a response or error.")

    def _should_retry(self, *, error: ClientError, attempt: int) -> bool:
        status_code = getattr(error, "code", None)
        if status_code != 429:
            return False
        if self._settings.llm_fail_fast_on_quota_exhaustion and self._is_non_transient_quota_error(error):
            return False
        return attempt < self._settings.llm_max_retries

    def _retry_delay_seconds(self, attempt: int, *, error: ClientError | None = None) -> float:
        base_delay = self._settings.llm_retry_base_delay_seconds
        max_delay = self._settings.llm_retry_max_delay_seconds
        default_delay = min(base_delay * (2**attempt), max_delay)
        suggested_delay = self._extract_retry_delay_seconds(error) if error is not None else None
        if suggested_delay is None:
            return default_delay
        return min(max(default_delay, suggested_delay), max_delay)

    def _wait_for_request_slot(self, model: str) -> None:
        min_interval = self._minimum_request_interval_seconds_for_model(model)
        if min_interval <= 0:
            return

        now = time.monotonic()
        next_request_not_before = self._next_request_not_before_by_model.get(model, 0.0)
        if next_request_not_before > now:
            time.sleep(next_request_not_before - now)
            now = time.monotonic()
        self._next_request_not_before_by_model[model] = now + min_interval

    def _minimum_request_interval_seconds_for_model(self, model: str) -> float:
        if self._settings.llm_min_request_interval_seconds is not None:
            return max(0.0, self._settings.llm_min_request_interval_seconds)

        normalized_model = model.strip().lower()
        if "gemma" in normalized_model:
            return 2.1
        if "2.5" in normalized_model and "flash-lite" in normalized_model:
            return 4.1
        if "2.5" in normalized_model and "flash" in normalized_model:
            return 6.1
        if "2.5" in normalized_model and "pro" in normalized_model:
            return 12.1
        if "2.0" in normalized_model and "flash-lite" in normalized_model:
            return 2.1
        if "2.0" in normalized_model and "flash" in normalized_model:
            return 4.1
        return 0.0

    def _extract_retry_delay_seconds(self, error: ClientError) -> float | None:
        details = getattr(error, "details", None)
        for retry_delay in self._find_retry_delay_values(details):
            parsed_retry_delay = self._parse_retry_delay_seconds(retry_delay)
            if parsed_retry_delay is not None:
                return parsed_retry_delay
        return None

    def _find_retry_delay_values(self, node: Any) -> list[Any]:
        found: list[Any] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"retryDelay", "retry_delay"}:
                    found.append(value)
                found.extend(self._find_retry_delay_values(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(self._find_retry_delay_values(item))
        return found

    def _parse_retry_delay_seconds(self, value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        if isinstance(value, str):
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", value)
            if match:
                return max(0.0, float(match.group(1)))
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
        return None

    def _is_non_transient_quota_error(self, error: ClientError) -> bool:
        lowered_message = str(getattr(error, "message", "") or str(error)).lower()
        quota_markers = (
            "per day",
            "daily",
            "quota",
            "billing",
            "limit 'generatecontentrequestsperday",
            "limit 'generatetokensperday",
            "free_tier",
        )
        return any(marker in lowered_message for marker in quota_markers)

    def _should_fallback_to_prompt_json(self, error: ClientError) -> bool:
        status_code = getattr(error, "code", None)
        if status_code != 400:
            return False
        lowered = str(error).lower()
        capability_markers = (
            "json mode is not enabled",
            "response schema",
            "response_json_schema",
            "developer instruction is not enabled",
            "system instruction is not enabled",
        )
        return any(marker in lowered for marker in capability_markers)

    def _build_prompt_json_request(
        self,
        *,
        prompt: str,
        system_instruction: str,
        json_schema: dict[str, Any],
    ) -> str:
        schema_json = json.dumps(json_schema, ensure_ascii=False, indent=2)
        return f"""
Follow these instructions exactly.

System instruction:
{system_instruction}

User request:
{prompt}

Return only valid JSON that matches this schema exactly:
{schema_json}
""".strip()

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
            if fenced_match:
                cleaned = fenced_match.group(1).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise RuntimeError("Google model returned text that did not contain a valid JSON object.")
            payload = json.loads(cleaned[start : end + 1])

        if not isinstance(payload, dict):
            raise RuntimeError("Google model returned JSON, but the top-level value was not an object.")
        return payload

    def _translate_client_error(self, error: ClientError) -> RuntimeError:
        status_code = getattr(error, "code", None)
        message = str(error)
        lowered = message.lower()

        if status_code == 403 and "reported as leaked" in lowered:
            return RuntimeError(
                "Google API key is blocked because Google flagged it as leaked. "
                "Create a new key in Google AI Studio, update `GOOGLE_API_KEY` or `GEMINI_API_KEY`, and retry."
            )

        if status_code == 403:
            return RuntimeError(
                "Google GenAI request was denied. Check that the API key is valid and has access to the selected model."
            )

        if status_code == 429:
            if self._settings.llm_fail_fast_on_quota_exhaustion and self._is_non_transient_quota_error(error):
                return RuntimeError(
                    "Google GenAI quota appears to be exhausted for the selected model. Waiting through retries "
                    "is unlikely to help. Try a lower-volume model, reduce dataset size, wait for quota reset, "
                    "or adjust `LLM_MIN_REQUEST_INTERVAL_SECONDS` / your Google plan. "
                    "For long runs, restart with `--resume-from-cache`."
                )
            return RuntimeError(
                "Google GenAI rate limit was hit repeatedly. Wait a bit, reduce request volume, "
                "or increase retry settings. For long runs, restart with `--resume-from-cache` "
                "to continue from already completed items."
            )

        return RuntimeError(f"Google GenAI request failed: {message}")


GeminiStructuredClient = GoogleGenAIStructuredClient
