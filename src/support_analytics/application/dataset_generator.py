from __future__ import annotations

from pathlib import Path
from typing import Callable

from support_analytics.application.ports import StructuredLLM
from support_analytics.application.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    build_generation_prompt,
)
from support_analytics.application.reports import build_dataset_manifest
from support_analytics.application.scenario_factory import ScenarioFactory
from support_analytics.domain.models import (
    ConversationArtifact,
    ConversationDraft,
    GenerationRequest,
    GroundTruthLabel,
    build_turns_from_draft,
)
from support_analytics.infrastructure.jsonl_repository import JsonlRepository
from support_analytics.infrastructure.replay_cache import ReplayCache

GenerationProgressCallback = Callable[[int, int, str, str], None]


class DatasetGeneratorService:
    def __init__(
        self,
        *,
        llm: StructuredLLM,
        repository: JsonlRepository,
        replay_cache: ReplayCache,
        scenario_factory: ScenarioFactory,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._replay_cache = replay_cache
        self._scenario_factory = scenario_factory

    def generate(
        self,
        *,
        request: GenerationRequest,
        model: str,
        progress_callback: GenerationProgressCallback | None = None,
    ) -> tuple[Path, Path]:
        blueprints = self._scenario_factory.build(
            count=request.count,
            seed=request.seed,
            language=request.language,
        )
        if request.start_from >= len(blueprints):
            raise RuntimeError(
                f"start_from={request.start_from} is out of range for {len(blueprints)} planned conversations."
            )

        output_path = Path(request.output_path)
        manifest_path = Path(request.manifest_path)
        expected_conversation_ids = [blueprint.conversation_id for blueprint in blueprints]
        conversations_by_id = self._load_existing_conversations(
            output_path=output_path,
            expected_conversation_ids=expected_conversation_ids,
            resume_from_cache=request.resume_from_cache,
            force_refresh=request.force_refresh,
        )

        total_to_process = len(blueprints) - request.start_from
        processed = 0

        for blueprint in blueprints[request.start_from :]:
            if request.resume_from_cache and blueprint.conversation_id in conversations_by_id:
                processed += 1
                self._notify_progress(
                    progress_callback=progress_callback,
                    completed=processed,
                    total=total_to_process,
                    conversation_id=blueprint.conversation_id,
                    status="reused-output",
                )
                continue

            prompt = build_generation_prompt(blueprint)
            cache_key = self._replay_cache.build_key(
                namespace="dataset_generation",
                model=model,
                seed=blueprint.generation_seed,
                system_instruction=GENERATOR_SYSTEM_PROMPT,
                prompt=prompt,
                schema_name=ConversationDraft.__name__,
            )
            draft = None
            was_loaded_from_cache = False
            if request.resume_from_cache and not request.force_refresh:
                draft = self._replay_cache.load(cache_key, ConversationDraft)
                was_loaded_from_cache = draft is not None
            if draft is None:
                draft = self._llm.generate_structured(
                    model=model,
                    prompt=prompt,
                    system_instruction=GENERATOR_SYSTEM_PROMPT,
                    response_schema=ConversationDraft,
                    seed=blueprint.generation_seed,
                )
                self._replay_cache.store(cache_key, draft)

            artifact = ConversationArtifact(
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
            conversations_by_id[artifact.conversation_id] = artifact
            self._repository.append_jsonl(output_path, artifact)
            self._write_manifest(
                manifest_path=manifest_path,
                output_path=output_path,
                conversations=self._ordered_conversations(
                    blueprints=blueprints,
                    conversations_by_id=conversations_by_id,
                ),
                seed=request.seed,
                language=request.language,
                model=model,
            )
            processed += 1
            self._notify_progress(
                progress_callback=progress_callback,
                completed=processed,
                total=total_to_process,
                conversation_id=blueprint.conversation_id,
                status="reused-cache" if was_loaded_from_cache else "generated",
            )

        ordered_conversations = self._ordered_conversations(
            blueprints=blueprints,
            conversations_by_id=conversations_by_id,
        )
        self._repository.write_jsonl(output_path, ordered_conversations)
        self._write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            conversations=ordered_conversations,
            seed=request.seed,
            language=request.language,
            model=model,
        )
        return output_path, manifest_path

    def _load_existing_conversations(
        self,
        *,
        output_path: Path,
        expected_conversation_ids: list[str],
        resume_from_cache: bool,
        force_refresh: bool,
    ) -> dict[str, ConversationArtifact]:
        if force_refresh or not resume_from_cache:
            self._repository.write_jsonl(output_path, [])
            return {}

        expected_ids = set(expected_conversation_ids)
        existing_conversations = self._repository.read_jsonl(output_path, ConversationArtifact)
        filtered_conversations = {
            conversation.conversation_id: conversation
            for conversation in existing_conversations
            if conversation.conversation_id in expected_ids
        }
        normalized_conversations = [
            filtered_conversations[conversation_id]
            for conversation_id in expected_conversation_ids
            if conversation_id in filtered_conversations
        ]
        self._repository.write_jsonl(output_path, normalized_conversations)
        return filtered_conversations

    def _ordered_conversations(
        self,
        *,
        blueprints: list,
        conversations_by_id: dict[str, ConversationArtifact],
    ) -> list[ConversationArtifact]:
        return [
            conversations_by_id[blueprint.conversation_id]
            for blueprint in blueprints
            if blueprint.conversation_id in conversations_by_id
        ]

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        output_path: Path,
        conversations: list[ConversationArtifact],
        seed: int,
        language: str,
        model: str,
    ) -> None:
        manifest = build_dataset_manifest(
            conversations=conversations,
            seed=seed,
            language=language,
            model=model,
            output_path=str(output_path),
        )
        self._repository.write_json(manifest_path, manifest)

    def _notify_progress(
        self,
        *,
        progress_callback: GenerationProgressCallback | None,
        completed: int,
        total: int,
        conversation_id: str,
        status: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(completed, total, conversation_id, status)
