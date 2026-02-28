from support_analytics.application.reports import build_evaluation_report
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
from support_analytics.domain.models import (
    AgentError,
    ChatTurn,
    ConversationArtifact,
    ConversationEvaluation,
    GroundTruthLabel,
    ScenarioBlueprint,
    SupportEvaluationRecord,
)
from support_analytics.domain.scoring import analyze_rule_signals


def _build_conversation() -> ConversationArtifact:
    blueprint = ScenarioBlueprint(
        conversation_id="chat_0042_001",
        language="uk",
        support_channel="live_chat",
        issue_type=IssueType.TECHNICAL_ERROR,
        case_class=CaseClass.AGENT_MISTAKE,
        customer_profile="Операційний менеджер, який блокується через критичну помилку.",
        scenario_prompt="Експорт звіту завершується помилкою 500.",
        customer_tone="напружено-ввічливий",
        resolution_status=ResolutionStatus.UNRESOLVED,
        visible_customer_satisfaction=CustomerSatisfaction.NEUTRAL,
        actual_customer_satisfaction=CustomerSatisfaction.DISSATISFIED,
        hidden_dissatisfaction=True,
        problem_solved=False,
        target_answer_quality_score=41,
        required_story_beats=[
            "клієнт описує технічну помилку",
            "агент робить помилковий висновок",
            "клієнт формально дякує наприкінці, але проблема лишається",
        ],
        required_agent_errors=[
            AgentError(
                type=AgentErrorType.LOGIC,
                description="агент робить поспішний висновок без діагностики",
                severity=4,
            )
        ],
        min_turns=6,
        max_turns=8,
        generation_seed=42,
    )
    return ConversationArtifact(
        conversation_id="chat_0042_001",
        language="uk",
        support_channel="live_chat",
        title="Помилка експорту після оновлення",
        scenario_summary="Клієнт не може завершити експорт звіту після оновлення застосунку.",
        turns=[
            ChatTurn(turn_index=0, role=ActorRole.CUSTOMER, message="Після оновлення експорт як і раніше не працює."),
            ChatTurn(turn_index=1, role=ActorRole.AGENT, message="У нас таких проблем немає, вам треба просто почекати."),
            ChatTurn(turn_index=2, role=ActorRole.CUSTOMER, message="Це вже вдруге. Я все ще не можу завершити експорт."),
            ChatTurn(turn_index=3, role=ActorRole.CUSTOMER, message="Добре, дякую, але проблема залишилась."),
        ],
        final_customer_signal="Клієнт ввічливо завершує чат, хоча явно не задоволений результатом.",
        blueprint=blueprint,
        ground_truth=GroundTruthLabel(
            issue_type=IssueType.TECHNICAL_ERROR,
            case_class=CaseClass.AGENT_MISTAKE,
            resolution_status=ResolutionStatus.UNRESOLVED,
            visible_customer_satisfaction=CustomerSatisfaction.NEUTRAL,
            actual_customer_satisfaction=CustomerSatisfaction.DISSATISFIED,
            hidden_dissatisfaction=True,
            problem_solved=False,
            answer_quality_score=41,
            required_agent_errors=blueprint.required_agent_errors,
        ),
    )


def test_rule_signals_detect_hidden_dissatisfaction() -> None:
    conversation = _build_conversation()
    signals = analyze_rule_signals(conversation)

    assert signals.hidden_dissatisfaction_hint is True
    assert signals.customer_frustration_count == 1
    assert signals.agent_certainty_risk_count == 1


def test_conversation_evaluation_deduplicates_agent_mistakes() -> None:
    evaluation = ConversationEvaluation(
        intent=SupportIntent.TECHNICAL_ERROR,
        satisfaction=SupportSatisfaction.UNSATISFIED,
        quality_score=2,
        agent_mistakes=[
            AgentMistakeTag.NO_RESOLUTION,
            AgentMistakeTag.NO_RESOLUTION,
            AgentMistakeTag.INCORRECT_INFO,
        ],
    )

    assert evaluation.agent_mistakes == [
        AgentMistakeTag.NO_RESOLUTION,
        AgentMistakeTag.INCORRECT_INFO,
    ]


def test_evaluation_report_aggregates_new_contract_fields() -> None:
    records = [
        SupportEvaluationRecord.from_evaluation(
            conversation_id="chat_001",
            evaluation=ConversationEvaluation(
                intent=SupportIntent.ACCOUNT_ACCESS,
                satisfaction=SupportSatisfaction.UNSATISFIED,
                quality_score=2,
                agent_mistakes=[AgentMistakeTag.NO_RESOLUTION, AgentMistakeTag.INCORRECT_INFO],
            ),
        ),
        SupportEvaluationRecord.from_evaluation(
            conversation_id="chat_002",
            evaluation=ConversationEvaluation(
                intent=SupportIntent.PLAN_QUESTION,
                satisfaction=SupportSatisfaction.SATISFIED,
                quality_score=5,
                agent_mistakes=[],
            ),
        ),
    ]

    report = build_evaluation_report(
        input_path="artifacts/datasets/support_chats.jsonl",
        output_path="artifacts/reports/support_evaluation.json",
        model="gemini-2.5-flash-lite",
        records=records,
    )

    assert report.average_quality_score == 3.5
    assert report.distribution_by_intent["account_access"] == 1
    assert report.distribution_by_intent["other"] == 0
    assert report.distribution_by_satisfaction["unsatisfied"] == 1
    assert report.distribution_by_quality_score["5"] == 1
    assert report.distribution_by_agent_mistake["no_resolution"] == 1
    assert report.distribution_by_agent_mistake["rude_tone"] == 0
