from __future__ import annotations

from collections import Counter
from statistics import mean

from support_analytics.domain.enums import AgentMistakeTag, SupportIntent, SupportSatisfaction
from support_analytics.domain.models import (
    ConversationArtifact,
    DatasetManifest,
    EvaluationReport,
    SupportEvaluationRecord,
)


def build_dataset_manifest(
    *,
    conversations: list[ConversationArtifact],
    seed: int,
    language: str,
    model: str,
    output_path: str,
) -> DatasetManifest:
    issue_distribution = Counter(conversation.ground_truth.issue_type for conversation in conversations)
    case_distribution = Counter(conversation.ground_truth.case_class for conversation in conversations)
    return DatasetManifest(
        total_conversations=len(conversations),
        seed=seed,
        language=language,
        model=model,
        output_path=output_path,
        distribution_by_issue={key.value: value for key, value in issue_distribution.items()},
        distribution_by_case_class={key.value: value for key, value in case_distribution.items()},
        hidden_dissatisfaction_cases=sum(
            1 for conversation in conversations if conversation.ground_truth.hidden_dissatisfaction
        ),
        agent_error_cases=sum(
            1 for conversation in conversations if conversation.ground_truth.required_agent_errors
        ),
    )


def build_evaluation_report(
    *,
    input_path: str,
    output_path: str,
    model: str,
    records: list[SupportEvaluationRecord],
) -> EvaluationReport:
    intent_distribution = Counter(record.intent for record in records)
    satisfaction_distribution = Counter(record.satisfaction for record in records)
    quality_distribution = Counter(record.quality_score for record in records)
    mistake_distribution = Counter(
        agent_mistake
        for record in records
        for agent_mistake in record.agent_mistakes
    )

    return EvaluationReport(
        input_path=input_path,
        output_path=output_path,
        model=model,
        total_conversations=len(records),
        average_quality_score=round(mean(record.quality_score for record in records), 2) if records else 0.0,
        distribution_by_intent={
            intent.value: intent_distribution.get(intent, 0) for intent in SupportIntent
        },
        distribution_by_satisfaction={
            satisfaction.value: satisfaction_distribution.get(satisfaction, 0)
            for satisfaction in SupportSatisfaction
        },
        distribution_by_quality_score={
            str(score): quality_distribution.get(score, 0) for score in range(1, 6)
        },
        distribution_by_agent_mistake={
            agent_mistake.value: mistake_distribution.get(agent_mistake, 0)
            for agent_mistake in AgentMistakeTag
        },
        records=records,
    )
