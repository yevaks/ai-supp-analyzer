from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from support_analytics.application.config import AppSettings
from support_analytics.application.dataset_generator import DatasetGeneratorService
from support_analytics.application.evaluator import SupportQualityEvaluatorService
from support_analytics.application.ports import StructuredLLM
from support_analytics.application.scenario_factory import ScenarioFactory
from support_analytics.domain.models import ConversationArtifact, EvaluationRequest, GenerationRequest
from support_analytics.infrastructure.jsonl_repository import JsonlRepository
from support_analytics.infrastructure.llm_factory import build_structured_llm
from support_analytics.infrastructure.replay_cache import ReplayCache


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Synthetic support chat dataset generator and support QA evaluator.",
)


def _build_dependencies() -> tuple[AppSettings, JsonlRepository, ReplayCache, StructuredLLM]:
    settings = AppSettings()
    repository = JsonlRepository()
    replay_cache = ReplayCache(settings.cache_dir)
    llm = build_structured_llm(settings)
    return settings, repository, replay_cache, llm


@app.command("generate")
def generate_dataset(
    count: int = typer.Option(40, min=20, max=500, help="Number of conversations to generate."),
    seed: int = typer.Option(42, min=0, help="Seed for deterministic blueprint planning and LLM generation."),
    language: str | None = typer.Option(None, help="Dataset language. Default is loaded from settings."),
    model: str | None = typer.Option(None, help="LLM model override."),
    output: Path = typer.Option(
        Path("artifacts/datasets/support_chats.jsonl"),
        help="Output JSONL dataset path.",
    ),
    manifest: Path = typer.Option(
        Path("artifacts/datasets/support_chats.manifest.json"),
        help="Output manifest path.",
    ),
    start_from: int = typer.Option(
        0,
        min=0,
        help="Zero-based conversation index to start from when resuming or generating a partial dataset.",
    ),
    resume_from_cache: bool = typer.Option(
        False,
        "--resume-from-cache/--no-resume-from-cache",
        help="Reuse existing output JSONL and replay-cache entries when resuming generation.",
    ),
    force_refresh: bool = typer.Option(
        False,
        help="Ignore replay cache and regenerate LLM outputs.",
    ),
) -> None:
    try:
        settings, repository, replay_cache, llm = _build_dependencies()
        generator = DatasetGeneratorService(
            llm=llm,
            repository=repository,
            replay_cache=replay_cache,
            scenario_factory=ScenarioFactory(),
        )
        request = GenerationRequest(
            count=count,
            seed=seed,
            language=language or settings.dataset_language,
            output_path=str(output),
            manifest_path=str(manifest),
            start_from=start_from,
            resume_from_cache=resume_from_cache,
            force_refresh=force_refresh,
        )
        model_name = settings.resolve_generation_model(model)
        progress_total = max(count - start_from, 0)
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Generating support chats", total=progress_total)

            def on_progress(completed: int, total: int, conversation_id: str, status: str) -> None:
                progress.update(
                    task_id,
                    total=total,
                    completed=completed,
                    description=f"Generating support chats [{status}] {conversation_id}",
                )

            dataset_path, manifest_path = generator.generate(
                request=request,
                model=model_name,
                progress_callback=on_progress,
            )
        typer.echo(f"Dataset written to {dataset_path}")
        typer.echo(f"Manifest written to {manifest_path}")
    except RuntimeError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command("evaluate")
def evaluate_dataset(
    input_path: Path = typer.Option(
        Path("artifacts/datasets/support_chats.jsonl"),
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input dataset JSONL path.",
    ),
    output: Path = typer.Option(
        Path("artifacts/reports/support_evaluation.json"),
        help="Output evaluation report path.",
    ),
    seed: int = typer.Option(42, min=0, help="Seed for deterministic LLM generation and evaluation."),
    model: str | None = typer.Option(None, help="LLM model override."),
    start_from: int = typer.Option(
        0,
        min=0,
        help="Zero-based conversation index to start from when resuming or evaluating a partial dataset.",
    ),
    resume_from_cache: bool = typer.Option(
        False,
        "--resume-from-cache/--no-resume-from-cache",
        help="Reuse existing output report and replay-cache entries when resuming evaluation.",
    ),
    force_refresh: bool = typer.Option(
        False,
        help="Ignore replay cache and rerun the evaluator prompts.",
    ),
) -> None:
    try:
        settings, repository, replay_cache, llm = _build_dependencies()
        evaluator = SupportQualityEvaluatorService(
            llm=llm,
            repository=repository,
            replay_cache=replay_cache,
        )
        request = EvaluationRequest(
            input_path=str(input_path),
            output_path=str(output),
            seed=seed,
            start_from=start_from,
            resume_from_cache=resume_from_cache,
            force_refresh=force_refresh,
        )
        model_name = settings.resolve_evaluation_model(model)
        total_conversations = len(repository.read_jsonl(input_path, ConversationArtifact))
        progress_total = max(total_conversations - start_from, 0)
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task("Evaluating support chats", total=progress_total)

            def on_progress(completed: int, total: int, conversation_id: str, status: str) -> None:
                progress.update(
                    task_id,
                    total=total,
                    completed=completed,
                    description=f"Evaluating support chats [{status}] {conversation_id}",
                )

            report_path = evaluator.evaluate(
                request=request,
                model=model_name,
                progress_callback=on_progress,
            )
        typer.echo(f"Evaluation report written to {report_path}")
    except RuntimeError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
