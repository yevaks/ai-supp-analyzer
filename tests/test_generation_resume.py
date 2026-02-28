from pathlib import Path

from support_analytics.application.dataset_generator import DatasetGeneratorService
from support_analytics.application.scenario_factory import ScenarioFactory
from support_analytics.domain.enums import ActorRole
from support_analytics.domain.models import (
    ConversationArtifact,
    ConversationDraft,
    DraftTurn,
    GenerationRequest,
    GroundTruthLabel,
    build_turns_from_draft,
)
from support_analytics.infrastructure.jsonl_repository import JsonlRepository
from support_analytics.infrastructure.replay_cache import ReplayCache


class FakeStructuredLLM:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: type[ConversationDraft],
        seed: int,
    ) -> ConversationDraft:
        self.calls.append(seed)
        return response_schema(
            title=f"Generated chat {seed}",
            scenario_summary=f"Synthetic summary for seed {seed}.",
            turns=[
                DraftTurn(role=ActorRole.CUSTOMER, message=f"Customer message {seed}-1"),
                DraftTurn(role=ActorRole.AGENT, message=f"Agent message {seed}-1"),
                DraftTurn(role=ActorRole.CUSTOMER, message=f"Customer message {seed}-2"),
                DraftTurn(role=ActorRole.AGENT, message=f"Agent message {seed}-2"),
            ],
            final_customer_signal=f"Signal {seed}",
        )


def _build_artifact(blueprint, seed: int) -> ConversationArtifact:
    draft = ConversationDraft(
        title=f"Generated chat {seed}",
        scenario_summary=f"Synthetic summary for seed {seed}.",
        turns=[
            DraftTurn(role=ActorRole.CUSTOMER, message=f"Customer message {seed}-1"),
            DraftTurn(role=ActorRole.AGENT, message=f"Agent message {seed}-1"),
            DraftTurn(role=ActorRole.CUSTOMER, message=f"Customer message {seed}-2"),
            DraftTurn(role=ActorRole.AGENT, message=f"Agent message {seed}-2"),
        ],
        final_customer_signal=f"Signal {seed}",
    )
    return ConversationArtifact(
        conversation_id=blueprint.conversation_id,
        language=blueprint.language,
        support_channel=blueprint.support_channel,
        title=draft.title,
        scenario_summary=draft.scenario_summary,
        turns=build_turns_from_draft(draft.turns),
        final_customer_signal=draft.final_customer_signal,
        blueprint=blueprint,
        ground_truth=GroundTruthLabel(
            issue_type=blueprint.issue_type,
            case_class=blueprint.case_class,
            resolution_status=blueprint.resolution_status,
            visible_customer_satisfaction=blueprint.visible_customer_satisfaction,
            actual_customer_satisfaction=blueprint.actual_customer_satisfaction,
            hidden_dissatisfaction=blueprint.hidden_dissatisfaction,
            problem_solved=blueprint.problem_solved,
            answer_quality_score=blueprint.target_answer_quality_score,
            required_agent_errors=blueprint.required_agent_errors,
        ),
    )


def _build_service(tmp_path: Path, llm: FakeStructuredLLM) -> tuple[DatasetGeneratorService, JsonlRepository]:
    repository = JsonlRepository()
    service = DatasetGeneratorService(
        llm=llm,
        repository=repository,
        replay_cache=ReplayCache(tmp_path / ".cache"),
        scenario_factory=ScenarioFactory(),
    )
    return service, repository


def test_generate_respects_start_from_and_writes_partial_dataset(tmp_path: Path) -> None:
    llm = FakeStructuredLLM()
    service, repository = _build_service(tmp_path, llm)
    factory = ScenarioFactory()
    blueprints = factory.build(count=20, seed=42, language="uk")
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    progress_updates: list[tuple[int, int, str, str]] = []

    service.generate(
        request=GenerationRequest(
            count=20,
            seed=42,
            language="uk",
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            start_from=2,
            resume_from_cache=True,
        ),
        model="gemini-2.5-flash-lite",
        progress_callback=lambda completed, total, conversation_id, status: progress_updates.append(
            (completed, total, conversation_id, status)
        ),
    )

    records = repository.read_jsonl(output_path, ConversationArtifact)
    assert len(records) == 18
    assert records[0].conversation_id == blueprints[2].conversation_id
    assert len(llm.calls) == 18
    assert progress_updates[0] == (1, 18, blueprints[2].conversation_id, "generated")
    assert progress_updates[-1][0:2] == (18, 18)


def test_generate_resumes_from_existing_output(tmp_path: Path) -> None:
    llm = FakeStructuredLLM()
    service, repository = _build_service(tmp_path, llm)
    blueprints = ScenarioFactory().build(count=20, seed=42, language="uk")
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    repository.write_jsonl(
        output_path,
        [_build_artifact(blueprint, blueprint.generation_seed) for blueprint in blueprints[:3]],
    )
    progress_updates: list[tuple[int, int, str, str]] = []

    service.generate(
        request=GenerationRequest(
            count=20,
            seed=42,
            language="uk",
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            resume_from_cache=True,
        ),
        model="gemini-2.5-flash-lite",
        progress_callback=lambda completed, total, conversation_id, status: progress_updates.append(
            (completed, total, conversation_id, status)
        ),
    )

    records = repository.read_jsonl(output_path, ConversationArtifact)
    assert len(records) == 20
    assert len(llm.calls) == 17
    assert [record.conversation_id for record in records[:3]] == [
        blueprint.conversation_id for blueprint in blueprints[:3]
    ]
    assert [status for _, _, _, status in progress_updates[:3]] == [
        "reused-output",
        "reused-output",
        "reused-output",
    ]
    assert any(status == "generated" for _, _, _, status in progress_updates[3:])


def test_generate_does_not_reuse_existing_output_without_flag(tmp_path: Path) -> None:
    llm = FakeStructuredLLM()
    service, repository = _build_service(tmp_path, llm)
    blueprints = ScenarioFactory().build(count=20, seed=42, language="uk")
    output_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "manifest.json"
    repository.write_jsonl(
        output_path,
        [_build_artifact(blueprint, blueprint.generation_seed) for blueprint in blueprints[:3]],
    )
    progress_updates: list[tuple[int, int, str, str]] = []

    service.generate(
        request=GenerationRequest(
            count=20,
            seed=42,
            language="uk",
            output_path=str(output_path),
            manifest_path=str(manifest_path),
        ),
        model="gemini-2.5-flash-lite",
        progress_callback=lambda completed, total, conversation_id, status: progress_updates.append(
            (completed, total, conversation_id, status)
        ),
    )

    records = repository.read_jsonl(output_path, ConversationArtifact)
    assert len(records) == 20
    assert len(llm.calls) == 20
    assert all(status == "generated" for _, _, _, status in progress_updates)
