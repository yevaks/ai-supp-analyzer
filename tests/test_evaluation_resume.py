from pathlib import Path

from support_analytics.application.evaluator import SupportQualityEvaluatorService
from support_analytics.domain.enums import (
    ActorRole,
    CaseClass,
    CustomerSatisfaction,
    IssueType,
    ResolutionStatus,
    SupportIntent,
    SupportSatisfaction,
)
from support_analytics.domain.models import (
    ChatTurn,
    ConversationArtifact,
    ConversationEvaluation,
    EvaluationReport,
    EvaluationRequest,
    GroundTruthLabel,
    ScenarioBlueprint,
)
from support_analytics.infrastructure.jsonl_repository import JsonlRepository
from support_analytics.infrastructure.replay_cache import ReplayCache


class FakeEvaluationLLM:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: type[ConversationEvaluation],
        seed: int,
    ) -> ConversationEvaluation:
        self.calls.append(seed)
        return response_schema(
            intent=SupportIntent.TECHNICAL_ERROR,
            satisfaction=SupportSatisfaction.UNSATISFIED,
            quality_score=2,
            agent_mistakes=[],
        )


def _build_conversation(conversation_id: str, seed: int) -> ConversationArtifact:
    blueprint = ScenarioBlueprint(
        conversation_id=conversation_id,
        language="uk",
        support_channel="live_chat",
        issue_type=IssueType.TECHNICAL_ERROR,
        case_class=CaseClass.PROBLEMATIC,
        customer_profile="Операційний менеджер, який звертається через збій експорту.",
        scenario_prompt="Експорт звіту завершується помилкою 500.",
        customer_tone="стривожений",
        resolution_status=ResolutionStatus.UNRESOLVED,
        visible_customer_satisfaction=CustomerSatisfaction.NEUTRAL,
        actual_customer_satisfaction=CustomerSatisfaction.DISSATISFIED,
        hidden_dissatisfaction=False,
        problem_solved=False,
        target_answer_quality_score=45,
        required_story_beats=[
            "клієнт повідомляє про технічну помилку",
            "агент ставить уточнювальні питання",
            "проблема лишається невирішеною",
        ],
        min_turns=4,
        max_turns=6,
        generation_seed=seed,
    )
    return ConversationArtifact(
        conversation_id=conversation_id,
        language="uk",
        support_channel="live_chat",
        title=f"Діалог {conversation_id}",
        scenario_summary="Клієнт звертається через технічну помилку в продукті.",
        turns=[
            ChatTurn(turn_index=0, role=ActorRole.CUSTOMER, message="Експорт не працює."),
            ChatTurn(turn_index=1, role=ActorRole.AGENT, message="Підкажіть, коли саме виникла помилка?"),
            ChatTurn(turn_index=2, role=ActorRole.CUSTOMER, message="Після оновлення."),
            ChatTurn(turn_index=3, role=ActorRole.AGENT, message="Передам технічній команді."),
        ],
        final_customer_signal="Клієнт лишається без рішення.",
        blueprint=blueprint,
        ground_truth=GroundTruthLabel(
            issue_type=IssueType.TECHNICAL_ERROR,
            case_class=CaseClass.PROBLEMATIC,
            resolution_status=ResolutionStatus.UNRESOLVED,
            visible_customer_satisfaction=CustomerSatisfaction.NEUTRAL,
            actual_customer_satisfaction=CustomerSatisfaction.DISSATISFIED,
            hidden_dissatisfaction=False,
            problem_solved=False,
            answer_quality_score=45,
            required_agent_errors=[],
        ),
    )


def _build_service(tmp_path: Path, llm: FakeEvaluationLLM) -> tuple[SupportQualityEvaluatorService, JsonlRepository]:
    repository = JsonlRepository()
    service = SupportQualityEvaluatorService(
        llm=llm,
        repository=repository,
        replay_cache=ReplayCache(tmp_path / ".cache"),
    )
    return service, repository


def test_evaluate_resumes_from_existing_output(tmp_path: Path) -> None:
    llm = FakeEvaluationLLM()
    service, repository = _build_service(tmp_path, llm)
    input_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "report.json"
    repository.write_jsonl(
        input_path,
        [_build_conversation("chat_001", 42), _build_conversation("chat_002", 43)],
    )
    repository.write_json(
        output_path,
        EvaluationReport(
            input_path=str(input_path),
            output_path=str(output_path),
            model="gemini-2.5-flash-lite",
            total_conversations=1,
            average_quality_score=4.0,
            distribution_by_intent={
                "payment_issue": 0,
                "technical_error": 1,
                "account_access": 0,
                "plan_question": 0,
                "refund_request": 0,
                "other": 0,
            },
            distribution_by_satisfaction={
                "satisfied": 1,
                "neutral": 0,
                "unsatisfied": 0,
            },
            distribution_by_quality_score={"1": 0, "2": 0, "3": 0, "4": 1, "5": 0},
            distribution_by_agent_mistake={
                "ignored_question": 0,
                "incorrect_info": 0,
                "rude_tone": 0,
                "no_resolution": 0,
                "unnecessary_escalation": 0,
            },
            records=[
                {
                    "conversation_id": "chat_001",
                    "intent": "technical_error",
                    "satisfaction": "satisfied",
                    "quality_score": 4,
                    "agent_mistakes": [],
                }
            ],
        ),
    )
    progress_updates: list[tuple[int, int, str, str]] = []

    service.evaluate(
        request=EvaluationRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            seed=42,
            resume_from_cache=True,
        ),
        model="gemini-2.5-flash-lite",
        progress_callback=lambda completed, total, conversation_id, status: progress_updates.append(
            (completed, total, conversation_id, status)
        ),
    )

    report = repository.read_json(output_path, EvaluationReport)
    assert report is not None
    assert len(report.records) == 2
    assert len(llm.calls) == 1
    assert progress_updates[0] == (1, 2, "chat_001", "reused-output")
    assert progress_updates[1][3] == "generated"


def test_evaluate_does_not_reuse_existing_output_without_flag(tmp_path: Path) -> None:
    llm = FakeEvaluationLLM()
    service, repository = _build_service(tmp_path, llm)
    input_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "report.json"
    repository.write_jsonl(
        input_path,
        [_build_conversation("chat_001", 42), _build_conversation("chat_002", 43)],
    )
    repository.write_json(
        output_path,
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model": "gemini-2.5-flash-lite",
            "total_conversations": 1,
            "average_quality_score": 5.0,
            "distribution_by_intent": {
                "payment_issue": 0,
                "technical_error": 1,
                "account_access": 0,
                "plan_question": 0,
                "refund_request": 0,
                "other": 0,
            },
            "distribution_by_satisfaction": {
                "satisfied": 1,
                "neutral": 0,
                "unsatisfied": 0,
            },
            "distribution_by_quality_score": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1},
            "distribution_by_agent_mistake": {
                "ignored_question": 0,
                "incorrect_info": 0,
                "rude_tone": 0,
                "no_resolution": 0,
                "unnecessary_escalation": 0,
            },
            "records": [],
        },
    )

    service.evaluate(
        request=EvaluationRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            seed=42,
        ),
        model="gemini-2.5-flash-lite",
    )

    report = repository.read_json(output_path, EvaluationReport)
    assert report is not None
    assert len(report.records) == 2
    assert len(llm.calls) == 2
    assert report.average_quality_score == 2.0
