from __future__ import annotations

from pathlib import Path
from typing import Callable

from support_analytics.application.ports import StructuredLLM
from support_analytics.application.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    build_evaluation_prompt,
)
from support_analytics.application.reports import build_evaluation_report
from support_analytics.domain.models import (
    ConversationArtifact,
    ConversationEvaluation,
    EvaluationReport,
    EvaluationRequest,
    SupportEvaluationRecord,
)
from support_analytics.infrastructure.jsonl_repository import JsonlRepository
from support_analytics.infrastructure.replay_cache import ReplayCache

EvaluationProgressCallback = Callable[[int, int, str, str], None]


class SupportQualityEvaluatorService:
    def __init__(
        self,
        *,
        llm: StructuredLLM,
        repository: JsonlRepository,
        replay_cache: ReplayCache,
    ) -> None:
        self._llm = llm
        self._repository = repository
        self._replay_cache = replay_cache

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        model: str,
        progress_callback: EvaluationProgressCallback | None = None,
    ) -> Path:
        input_path = Path(request.input_path)
        output_path = Path(request.output_path)
        conversations = self._repository.read_jsonl(input_path, ConversationArtifact)
        if not conversations:
            self._repository.write_json(
                output_path,
                build_evaluation_report(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    model=model,
                    records=[],
                ),
            )
            return output_path
        if request.start_from >= len(conversations):
            raise RuntimeError(
                f"start_from={request.start_from} is out of range for {len(conversations)} input conversations."
            )

        expected_conversation_ids = [conversation.conversation_id for conversation in conversations]
        records_by_id = self._load_existing_records(
            output_path=output_path,
            input_path=input_path,
            expected_conversation_ids=expected_conversation_ids,
            model=model,
            resume_from_cache=request.resume_from_cache,
            force_refresh=request.force_refresh,
        )
        total_to_process = len(conversations) - request.start_from
        processed = 0

        for index, conversation in enumerate(conversations[request.start_from :], start=request.start_from):
            if request.resume_from_cache and conversation.conversation_id in records_by_id:
                processed += 1
                self._notify_progress(
                    progress_callback=progress_callback,
                    completed=processed,
                    total=total_to_process,
                    conversation_id=conversation.conversation_id,
                    status="reused-output",
                )
                continue

            prompt = build_evaluation_prompt(conversation)
            current_seed = request.seed + index
            cache_key = self._replay_cache.build_key(
                namespace="support_evaluation",
                model=model,
                seed=current_seed,
                system_instruction=EVALUATOR_SYSTEM_PROMPT,
                prompt=prompt,
                schema_name=ConversationEvaluation.__name__,
            )
            evaluation = None
            was_loaded_from_cache = False
            if request.resume_from_cache and not request.force_refresh:
                evaluation = self._replay_cache.load(cache_key, ConversationEvaluation)
                was_loaded_from_cache = evaluation is not None
            if evaluation is None:
                evaluation = self._llm.generate_structured(
                    model=model,
                    prompt=prompt,
                    system_instruction=EVALUATOR_SYSTEM_PROMPT,
                    response_schema=ConversationEvaluation,
                    seed=current_seed,
                )
                self._replay_cache.store(cache_key, evaluation)

            records_by_id[conversation.conversation_id] = SupportEvaluationRecord.from_evaluation(
                conversation_id=conversation.conversation_id,
                evaluation=evaluation,
            )
            self._write_report(
                output_path=output_path,
                input_path=input_path,
                model=model,
                conversations=conversations,
                records_by_id=records_by_id,
            )
            processed += 1
            self._notify_progress(
                progress_callback=progress_callback,
                completed=processed,
                total=total_to_process,
                conversation_id=conversation.conversation_id,
                status="reused-cache" if was_loaded_from_cache else "generated",
            )

        self._write_report(
            output_path=output_path,
            input_path=input_path,
            model=model,
            conversations=conversations,
            records_by_id=records_by_id,
        )
        return output_path

    def _load_existing_records(
        self,
        *,
        output_path: Path,
        input_path: Path,
        expected_conversation_ids: list[str],
        model: str,
        resume_from_cache: bool,
        force_refresh: bool,
    ) -> dict[str, SupportEvaluationRecord]:
        if force_refresh or not resume_from_cache:
            self._repository.write_json(
                output_path,
                build_evaluation_report(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    model=model,
                    records=[],
                ),
            )
            return {}

        existing_report = self._repository.read_json(output_path, EvaluationReport)
        if existing_report is None:
            return {}

        expected_ids = set(expected_conversation_ids)
        filtered_records = {
            record.conversation_id: record
            for record in existing_report.records
            if record.conversation_id in expected_ids
        }
        normalized_records = [
            filtered_records[conversation_id]
            for conversation_id in expected_conversation_ids
            if conversation_id in filtered_records
        ]
        self._repository.write_json(
            output_path,
            build_evaluation_report(
                input_path=str(input_path),
                output_path=str(output_path),
                model=model,
                records=normalized_records,
            ),
        )
        return filtered_records

    def _write_report(
        self,
        *,
        output_path: Path,
        input_path: Path,
        model: str,
        conversations: list[ConversationArtifact],
        records_by_id: dict[str, SupportEvaluationRecord],
    ) -> None:
        ordered_records = [
            records_by_id[conversation.conversation_id]
            for conversation in conversations
            if conversation.conversation_id in records_by_id
        ]
        report = build_evaluation_report(
            input_path=str(input_path),
            output_path=str(output_path),
            model=model,
            records=ordered_records,
        )
        self._repository.write_json(output_path, report)

    def _notify_progress(
        self,
        *,
        progress_callback: EvaluationProgressCallback | None,
        completed: int,
        total: int,
        conversation_id: str,
        status: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(completed, total, conversation_id, status)
