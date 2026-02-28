from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from support_analytics.domain.enums import (
    ActorRole,
    AgentErrorType,
    AgentMistakeTag,
    CaseClass,
    CustomerSatisfaction,
    IssueType,
    ResolutionStatus,
    SupportIntent,
    SupportSatisfaction,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DraftTurn(StrictModel):
    role: ActorRole
    message: str = Field(min_length=1, max_length=1000)


class ChatTurn(StrictModel):
    turn_index: int = Field(ge=0)
    role: ActorRole
    message: str = Field(min_length=1, max_length=1000)


class AgentError(StrictModel):
    type: AgentErrorType
    description: str = Field(min_length=3, max_length=240)
    severity: int = Field(ge=1, le=5)


class ScenarioBlueprint(StrictModel):
    conversation_id: str
    language: str = Field(default="uk", min_length=2, max_length=8)
    support_channel: str = Field(min_length=2, max_length=40)
    issue_type: IssueType
    case_class: CaseClass
    customer_profile: str = Field(min_length=8, max_length=240)
    scenario_prompt: str = Field(min_length=12, max_length=500)
    customer_tone: str = Field(min_length=3, max_length=40)
    resolution_status: ResolutionStatus
    visible_customer_satisfaction: CustomerSatisfaction
    actual_customer_satisfaction: CustomerSatisfaction
    hidden_dissatisfaction: bool
    problem_solved: bool
    target_answer_quality_score: int = Field(ge=0, le=100)
    required_story_beats: list[str] = Field(min_length=3, max_length=8)
    required_agent_errors: list[AgentError] = Field(default_factory=list)
    min_turns: int = Field(default=6, ge=4, le=20)
    max_turns: int = Field(default=10, ge=4, le=20)
    generation_seed: int = Field(ge=0)

    @computed_field(return_type=str)
    @property
    def blueprint_signature(self) -> str:
        payload = self.model_dump_json(exclude={"blueprint_signature"}, exclude_none=True)
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


class GroundTruthLabel(StrictModel):
    issue_type: IssueType
    case_class: CaseClass
    resolution_status: ResolutionStatus
    visible_customer_satisfaction: CustomerSatisfaction
    actual_customer_satisfaction: CustomerSatisfaction
    hidden_dissatisfaction: bool
    problem_solved: bool
    answer_quality_score: int = Field(ge=0, le=100)
    required_agent_errors: list[AgentError] = Field(default_factory=list)

    @computed_field(return_type=bool)
    @property
    def contains_agent_tone_error(self) -> bool:
        return any(error.type == AgentErrorType.TONE for error in self.required_agent_errors)

    @computed_field(return_type=bool)
    @property
    def contains_agent_logic_error(self) -> bool:
        return any(error.type == AgentErrorType.LOGIC for error in self.required_agent_errors)


class ConversationDraft(StrictModel):
    title: str = Field(min_length=8, max_length=120)
    scenario_summary: str = Field(min_length=20, max_length=320)
    turns: list[DraftTurn] = Field(min_length=4, max_length=20)
    final_customer_signal: str = Field(min_length=6, max_length=200)


class ConversationArtifact(StrictModel):
    conversation_id: str
    language: str
    support_channel: str
    title: str
    scenario_summary: str
    turns: list[ChatTurn] = Field(min_length=4, max_length=20)
    final_customer_signal: str
    blueprint: ScenarioBlueprint
    ground_truth: GroundTruthLabel


class RuleSignals(StrictModel):
    customer_gratitude_count: int = Field(default=0, ge=0)
    customer_frustration_count: int = Field(default=0, ge=0)
    unresolved_customer_signal_count: int = Field(default=0, ge=0)
    agent_apology_count: int = Field(default=0, ge=0)
    agent_certainty_risk_count: int = Field(default=0, ge=0)
    hidden_dissatisfaction_hint: bool = False
    explicit_conflict_hint: bool = False


class ConversationEvaluation(StrictModel):
    intent: SupportIntent
    satisfaction: SupportSatisfaction
    quality_score: int = Field(ge=1, le=5)
    agent_mistakes: list[AgentMistakeTag] = Field(default_factory=list, max_length=5)

    @field_validator("agent_mistakes")
    @classmethod
    def deduplicate_agent_mistakes(
        cls,
        agent_mistakes: list[AgentMistakeTag],
    ) -> list[AgentMistakeTag]:
        deduplicated: list[AgentMistakeTag] = []
        for agent_mistake in agent_mistakes:
            if agent_mistake not in deduplicated:
                deduplicated.append(agent_mistake)
        return deduplicated


class SupportEvaluationRecord(StrictModel):
    conversation_id: str
    intent: SupportIntent
    satisfaction: SupportSatisfaction
    quality_score: int = Field(ge=1, le=5)
    agent_mistakes: list[AgentMistakeTag] = Field(default_factory=list, max_length=5)

    @classmethod
    def from_evaluation(
        cls,
        *,
        conversation_id: str,
        evaluation: ConversationEvaluation,
    ) -> "SupportEvaluationRecord":
        return cls(
            conversation_id=conversation_id,
            intent=evaluation.intent,
            satisfaction=evaluation.satisfaction,
            quality_score=evaluation.quality_score,
            agent_mistakes=evaluation.agent_mistakes,
        )


class DatasetManifest(StrictModel):
    total_conversations: int = Field(ge=0)
    seed: int = Field(ge=0)
    language: str
    model: str
    output_path: str
    distribution_by_issue: dict[str, int]
    distribution_by_case_class: dict[str, int]
    hidden_dissatisfaction_cases: int = Field(ge=0)
    agent_error_cases: int = Field(ge=0)


class EvaluationReport(StrictModel):
    input_path: str
    output_path: str
    model: str
    total_conversations: int = Field(ge=0)
    average_quality_score: float = Field(ge=0, le=5)
    distribution_by_intent: dict[str, int]
    distribution_by_satisfaction: dict[str, int]
    distribution_by_quality_score: dict[str, int]
    distribution_by_agent_mistake: dict[str, int]
    records: list[SupportEvaluationRecord]


class GenerationRequest(StrictModel):
    count: int = Field(ge=20, le=500)
    seed: int = Field(default=42, ge=0)
    language: str = Field(default="uk", min_length=2, max_length=8)
    output_path: str
    manifest_path: str
    start_from: int = Field(default=0, ge=0)
    resume_from_cache: bool = False
    force_refresh: bool = False


class EvaluationRequest(StrictModel):
    input_path: str
    output_path: str
    seed: int = Field(default=42, ge=0)
    start_from: int = Field(default=0, ge=0)
    resume_from_cache: bool = False
    force_refresh: bool = False


def build_turns_from_draft(turns: list[DraftTurn]) -> list[ChatTurn]:
    return [
        ChatTurn(turn_index=index, role=turn.role, message=turn.message.strip())
        for index, turn in enumerate(turns)
    ]


def normalize_json_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")
