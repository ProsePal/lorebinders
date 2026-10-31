"""Unit tests for the workflow module."""

import asyncio
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lorebinders import models
from lorebinders.models import ChapterAppearances
from lorebinders.settings import Settings, get_settings
from lorebinders.storage import StorageProvider
from lorebinders.workflow import (
    _aggregate_to_binder,
    build_binder,
    extraction_categories,
    merge_traits,
)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary workspace directory."""
    ws = tmp_path / "work"
    ws.mkdir()
    os.environ["LOREBINDERS_WORKSPACE_BASE_PATH"] = str(ws)
    get_settings.cache_clear()
    return ws


@pytest.fixture
def book_file(tmp_path: Path) -> Path:
    """Fixture providing a temporary book file."""
    path = tmp_path / "book.txt"
    path.write_text("Chapter 1\nAlice.")
    return path


@pytest.fixture
def run_config(book_file: Path) -> models.RunConfiguration:
    """Fixture providing a standard run configuration."""
    return models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=book_file, title="Test Book")],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )


def _make_fake_book() -> models.Book:
    return models.Book(
        title="Test Book",
        author="Test Author",
        chapters=[
            models.Chapter(number=1, title="Ch1", content="Alice content")
        ],
    )


def test_aggregate_to_binder_structure() -> None:
    """Test that profiles are aggregated into binder structure."""
    profiles = [
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=1,
            book_title="Book 1",
            traits={"Role": "Hero"},
        ),
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=2,
            book_title="Book 1",
            traits={"Age": "20"},
        ),
    ]

    binder = _aggregate_to_binder(profiles, tracking="nested")

    assert "Characters" in binder.categories
    alice = binder.categories["Characters"].entities["Alice"]
    book_app = alice.appearances["Book 1"]
    assert isinstance(book_app, ChapterAppearances)
    assert book_app.chapters[1].traits == {"Role": "Hero"}
    assert book_app.chapters[2].traits == {"Age": "20"}


def test_merge_traits_includes_allusion_traits() -> None:
    """Test that allusions use their configured default traits."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )

    traits = merge_traits(Settings(), config)

    assert traits["Allusions"] == [
        "Invoked by",
        "Rhetorical significance",
        "What it reveals about the invoker",
    ]


def test_extraction_categories_excludes_allusions() -> None:
    """Test that Allusions is never sent to the extraction agent."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )

    categories = extraction_categories(Settings(), config)

    assert categories == ["Characters", "Locations"]
    assert "Allusions" not in categories


def test_extraction_categories_includes_custom_categories_and_traits() -> None:
    """Test that custom categories and custom trait keys are included."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
        custom_categories=["Factions"],
        custom_traits={"Objects": ["Origin"]},
    )

    categories = extraction_categories(Settings(), config)

    assert categories == ["Characters", "Locations", "Factions", "Objects"]


@pytest.mark.anyio
async def test_build_binder_orchestration(
    temp_workspace: Path,
    run_config: models.RunConfiguration,
) -> None:
    """Test end-to-end binder build orchestration with patched collaborators."""
    fake_book = _make_fake_book()
    fake_profiles = [
        models.EntityProfile(
            name="Alice",
            category="Characters",
            chapter_number=1,
            book_title="Test Book",
            traits={"Role": "Hero"},
        )
    ]

    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Series"
    fake_storage.extraction_exists.return_value = False
    fake_storage.profile_exists.return_value = False

    with (
        patch(
            "lorebinders.workflow.convert_to_text",
            return_value="Chapter 1\nAlice content",
        ) as mock_convert,
        patch(
            "lorebinders.workflow.ingest", return_value=fake_book
        ) as mock_ingest,
        patch(
            "lorebinders.workflow.extract_book",
            new_callable=AsyncMock,
            return_value={
                1: {
                    "Characters": [
                        models.ExtractedEntity(
                            name="Alice", presence_type="literal_entity"
                        )
                    ]
                }
            },
        ) as mock_extract,
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=fake_profiles,
        ),
        patch(
            "lorebinders.workflow.summarize_binder",
            new_callable=AsyncMock,
        ),
        patch("lorebinders.workflow.generate_pdf_report") as mock_report,
    ):
        result = await build_binder(
            run_config, provider=MagicMock(return_value=fake_storage)
        )

    mock_convert.assert_called_once_with(run_config.books[0].path)
    mock_ingest.assert_called_once_with(
        "Chapter 1\nAlice content", run_config.books[0].title
    )
    extraction_call_categories = mock_extract.call_args.args[3]
    assert "Allusions" not in extraction_call_categories
    mock_report.assert_called_once()
    assert (
        result
        == temp_workspace
        / "Test_Author"
        / "Test_Series"
        / "Test_Series_story_bible.pdf"
    )


@pytest.mark.anyio
async def test_build_binder_forwards_user_id(
    temp_workspace: Path,
    book_file: Path,
) -> None:
    """Test that config.user_id is forwarded to storage.set_workspace."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[models.BookInput(path=book_file, title="Test Book")],
        author_name="Test Author",
        user_id="user_123",
        narrator_config=models.NarratorConfig(),
    )
    fake_book = _make_fake_book()
    fake_storage = MagicMock()
    fake_storage.path = (
        temp_workspace / "user_123" / "Test_Author" / "Test_Series"
    )

    with (
        patch("lorebinders.workflow.convert_to_text", return_value=""),
        patch("lorebinders.workflow.ingest", return_value=fake_book),
        patch(
            "lorebinders.workflow.extract_book",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("lorebinders.workflow.summarize_binder", new_callable=AsyncMock),
        patch("lorebinders.workflow.generate_pdf_report"),
    ):
        await build_binder(
            config, provider=MagicMock(return_value=fake_storage)
        )

    workspace_args, workspace_kwargs = fake_storage.set_workspace.call_args
    assert workspace_args[0] == "Test Author"
    assert workspace_args[1] == "Test Series"
    assert workspace_kwargs["user_id"] == "user_123"
    assert len(workspace_kwargs["cache_namespace"]) == 64


@pytest.mark.anyio
async def test_build_binder_accumulates_stage_failures(
    temp_workspace: Path,
    book_file: Path,
) -> None:
    """Test that stage failures correctly accumulate across multiple books."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[
            models.BookInput(path=book_file, title="Book 1"),
            models.BookInput(path=book_file, title="Book 2"),
        ],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )
    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Series"

    from typing import Any

    def mock_ingest_func(
        content: str, title: str, *args: Any, **kwargs: Any
    ) -> models.Book:
        book = _make_fake_book()
        book.title = title
        return book

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[Any, Any]:
        book = args[0]
        on_obs = kwargs.get("on_observe") or args[7]
        failed = 1 if book.title == "Book 1" else 0
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="extraction",
                message=f"{failed} failed",
                metadata={
                    "failed_count": failed,
                    "total_count": 5,
                    "stage": "extraction",
                },
            )
        )
        return {}

    async def mock_analyze(*args: Any, **kwargs: Any) -> list[Any]:
        book = args[1]
        on_obs = kwargs.get("on_observe") or args[7]
        failed = 2 if book.title == "Book 1" else 0
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="analysis",
                message=f"{failed} failed",
                metadata={
                    "failed_count": failed,
                    "total_count": 10,
                    "stage": "analysis",
                },
            )
        )
        return []

    async def mock_summarize(*args: Any, **kwargs: Any) -> None:
        on_obs = kwargs.get("on_observe") or args[5]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="summarization",
                message="failed",
                metadata={
                    "failed_count": 3,
                    "total_count": 15,
                    "stage": "summarization",
                },
            )
        )
        return None

    with (
        patch("lorebinders.workflow.convert_to_text", return_value=""),
        patch("lorebinders.workflow.ingest", side_effect=mock_ingest_func),
        patch("lorebinders.workflow.extract_book", side_effect=mock_extract),
        patch(
            "lorebinders.workflow.analyze_entities", side_effect=mock_analyze
        ),
        patch(
            "lorebinders.workflow.summarize_binder", side_effect=mock_summarize
        ),
        patch("lorebinders.workflow.generate_pdf_report") as mock_report,
    ):
        await build_binder(
            config, provider=MagicMock(return_value=fake_storage)
        )

    binder = mock_report.call_args.args[0]

    assert binder.stage_failures["extraction"]["failed_count"] == 1
    assert binder.stage_failures["extraction"]["total_count"] == 10

    assert binder.stage_failures["analysis"]["failed_count"] == 2
    assert binder.stage_failures["analysis"]["total_count"] == 20

    assert binder.stage_failures["summarization"]["failed_count"] == 3
    assert binder.stage_failures["summarization"]["total_count"] == 15


@pytest.mark.anyio
async def test_build_binder_clean_run_no_partial_results(
    temp_workspace: Path,
    book_file: Path,
) -> None:
    """Test a clean run generates a PDF without the partial results note."""
    config = models.RunConfiguration(
        series_title="Test Series",
        books=[
            models.BookInput(path=book_file, title="Book 1"),
        ],
        author_name="Test Author",
        narrator_config=models.NarratorConfig(),
    )
    fake_storage = MagicMock()
    fake_storage.path = temp_workspace / "Test_Author" / "Test_Series"
    fake_storage.path.mkdir(parents=True, exist_ok=True)

    from typing import Any

    from pypdf import PdfReader

    def mock_ingest_func(
        content: str, title: str, *args: Any, **kwargs: Any
    ) -> models.Book:
        book = _make_fake_book()
        book.title = title
        return book

    async def mock_extract(*args: Any, **kwargs: Any) -> dict[Any, Any]:
        on_obs = kwargs.get("on_observe") or args[7]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="extraction",
                message="0 failed",
                metadata={
                    "failed_count": 0,
                    "total_count": 5,
                    "stage": "extraction",
                },
            )
        )
        return {}

    async def mock_analyze(*args: Any, **kwargs: Any) -> list[Any]:
        on_obs = kwargs.get("on_observe") or args[7]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="analysis",
                message="0 failed",
                metadata={
                    "failed_count": 0,
                    "total_count": 10,
                    "stage": "analysis",
                },
            )
        )
        return []

    async def mock_summarize(*args: Any, **kwargs: Any) -> None:
        on_obs = kwargs.get("on_observe") or args[5]
        on_obs(
            models.ObservationEvent(
                type=models.ObservationType.METRIC,
                stage="summarization",
                message="0 failed",
                metadata={
                    "failed_count": 0,
                    "total_count": 15,
                    "stage": "summarization",
                },
            )
        )
        return None

    with (
        patch("lorebinders.workflow.convert_to_text", return_value=""),
        patch("lorebinders.workflow.ingest", side_effect=mock_ingest_func),
        patch("lorebinders.workflow.extract_book", side_effect=mock_extract),
        patch(
            "lorebinders.workflow.analyze_entities", side_effect=mock_analyze
        ),
        patch(
            "lorebinders.workflow.summarize_binder", side_effect=mock_summarize
        ),
    ):
        result_pdf_path = await build_binder(
            config, provider=MagicMock(return_value=fake_storage)
        )

    assert result_pdf_path.exists()

    reader = PdfReader(result_pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    assert "Note: Partial Results" not in text


@pytest.mark.anyio
async def test_runtime_prompts_reach_agent_dependencies(
    run_config: models.RunConfiguration, temp_workspace: Path, tmp_path: Path
) -> None:
    prompts = {}
    for stage in (
        "extraction",
        "analysis",
        "summarization",
        "alias_resolution",
    ):
        path = tmp_path / f"{stage}.txt"
        path.write_text(f"Custom {stage}", encoding="utf-8")
        prompts[stage] = path

    class ObservedDependenciesError(Exception):
        pass

    async def inspect_extraction(
        book: models.Book,
        agent: Any,
        deps: models.AgentDeps,
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        for stage in prompts:
            assert deps.prompt_loader(f"{stage}.txt") == f"Custom {stage}"
        raise ObservedDependenciesError

    with patch(
        "lorebinders.workflow.extract_book", side_effect=inspect_extraction
    ):
        with pytest.raises(ObservedDependenciesError):
            await build_binder(run_config, prompt_files=prompts)


@pytest.mark.anyio
async def test_invalid_prompt_fails_before_agent_construction(
    run_config: models.RunConfiguration, tmp_path: Path
) -> None:
    with patch(
        "lorebinders.workflow.create_extraction_agent",
        side_effect=AssertionError("Agent constructed too soon"),
    ):
        with pytest.raises(FileNotFoundError):
            await build_binder(
                run_config,
                prompt_files={"alias_resolution": tmp_path / "missing"},
            )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changed",
    [
        "default_prompt",
        "analysis_prompt",
        "summarization_prompt",
        "alias_resolution_prompt",
        "override_prompt",
        "narrator",
        "model",
        "input",
        "traits",
        "categories",
        "appearance",
        "model_settings",
        "wrapped_settings",
    ],
)
async def test_workspace_cache_identity_tracks_effective_inputs(
    run_config: models.RunConfiguration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    from lorebinders.agent.extraction import extract_book as real_extract_book
    from lorebinders.prompts import build_prompt_loader
    from lorebinders.storage.factory import get_storage as real_get_storage

    settings = Settings(
        workspace_base_path=tmp_path / "cache", alias_resolution_enabled=False
    )
    monkeypatch.setattr("lorebinders.workflow.get_settings", lambda: settings)
    monkeypatch.setattr(
        "lorebinders.storage.workspace.get_settings", lambda: settings
    )
    real_get_storage.cache_clear()
    wrapped_temperature = [0.1]
    if changed == "wrapped_settings":
        from lorebinders.agent.factory import (
            ConfiguredModel,
            create_extraction_agent,
        )

        def configured_extraction(
            settings: Settings,
        ) -> Agent[models.AgentDeps, models.ExtractionResult]:
            agent = create_extraction_agent(settings)
            agent.model = ConfiguredModel(
                TestModel(), {"temperature": wrapped_temperature[0]}
            )
            return agent

        monkeypatch.setattr(
            "lorebinders.workflow.create_extraction_agent",
            configured_extraction,
        )
    prompt_path = tmp_path / "override.txt"
    prompt_path.write_text("Original prompt")
    prompt_files = (
        {"extraction": prompt_path} if changed == "override_prompt" else None
    )
    visits: list[Path] = []
    observations: list[models.ObservationEvent] = []

    async def cached_extraction(
        book: models.Book,
        agent: Agent[models.AgentDeps, models.ExtractionResult],
        deps: models.AgentDeps,
        categories: list[str],
        config: models.RunConfiguration,
        storage: StorageProvider,
        progress: Callable[[models.ProgressUpdate], None] | None = None,
        on_observe: Callable[[models.ObservationEvent], None] | None = None,
    ) -> dict[int, dict[str, list[models.ExtractedEntity]]]:
        visits.append(storage.path)
        with agent.override(
            model=TestModel(custom_output_args={"results": []})
        ):
            return await real_extract_book(
                book,
                agent,
                deps,
                categories,
                config,
                storage,
                progress,
                on_observe,
            )

    with (
        patch(
            "lorebinders.workflow.extract_book", side_effect=cached_extraction
        ),
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("lorebinders.workflow.summarize_binder", new_callable=AsyncMock),
        patch("lorebinders.workflow.generate_pdf_report"),
    ):
        await build_binder(
            run_config,
            prompt_files=prompt_files,
            on_observe=observations.append,
        )
        await build_binder(
            run_config,
            prompt_files=prompt_files,
            on_observe=observations.append,
        )
        if changed in {
            "default_prompt",
            "analysis_prompt",
            "summarization_prompt",
            "alias_resolution_prompt",
        }:
            original = build_prompt_loader()

            def revised_loader(
                files: Mapping[str, Path] | None,
            ) -> Callable[[str], str]:
                def load(filename: str) -> str:
                    return original(filename) + (
                        " updated"
                        if filename
                        == (
                            "extraction.txt"
                            if changed == "default_prompt"
                            else changed.removesuffix("_prompt") + ".txt"
                        )
                        else ""
                    )

                return load

            monkeypatch.setattr(
                "lorebinders.workflow.build_prompt_loader", revised_loader
            )
        elif changed == "override_prompt":
            prompt_path.write_text("Updated prompt")
        elif changed == "narrator":
            run_config.narrator_config = models.NarratorConfig(
                is_1st_person=True, name="Kalia"
            )
        elif changed == "model":
            settings.extraction_model = "openrouter:test/new-model"
        elif changed == "input":
            run_config.books[0].path.write_text(
                "Chapter 1\nA different manuscript."
            )
        elif changed == "traits":
            run_config.custom_traits = {"Characters": ["Motivation"]}
        elif changed == "categories":
            run_config.custom_categories = ["Organizations"]
        elif changed == "appearance":
            run_config.appearance_tracking = "flat"
        elif changed == "model_settings":
            settings.max_output_tokens = 1234
        elif changed == "wrapped_settings":
            wrapped_temperature[0] = 0.7
        output = await build_binder(
            run_config,
            prompt_files=prompt_files,
            on_observe=observations.append,
        )

    assert visits[0] == visits[1]
    assert visits[2] != visits[0]
    assert (
        sum(
            event.type == models.ObservationType.AGENT_RUN_STARTED
            for event in observations
        )
        == 2
    )
    assert output.name == "Test_Series_story_bible.pdf"
    real_get_storage.cache_clear()


def test_cache_fingerprint_accepts_stable_timeout_and_path_settings(
    run_config: models.RunConfiguration,
) -> None:
    import httpx

    from lorebinders.workflow import _cache_namespace

    def loader(filename: str) -> str:
        return "prompt"

    first = _cache_namespace(
        run_config,
        Settings(),
        loader,
        [{"timeout": httpx.Timeout(30), "path": Path("/operator/path")}],
    )
    same = _cache_namespace(
        run_config,
        Settings(),
        loader,
        [{"timeout": httpx.Timeout(30), "path": Path("/operator/path")}],
    )
    changed = _cache_namespace(
        run_config,
        Settings(),
        loader,
        [{"timeout": httpx.Timeout(31), "path": Path("/operator/path")}],
    )
    assert first == same
    assert first != changed


def test_cache_fingerprint_does_not_reuse_unknown_setting_objects(
    run_config: models.RunConfiguration,
) -> None:
    from lorebinders.workflow import _cache_namespace

    def loader(filename: str) -> str:
        return "prompt"

    value = object()
    first = _cache_namespace(
        run_config, Settings(), loader, [{"unknown": value}]
    )
    second = _cache_namespace(
        run_config, Settings(), loader, [{"unknown": value}]
    )
    assert first != second


@pytest.mark.anyio
async def test_concurrent_builds_keep_independent_storage(
    run_config: models.RunConfiguration,
    temp_workspace: Path,
) -> None:
    ready = asyncio.Event()
    visits: list[tuple[StorageProvider, Path, str]] = []

    async def concurrent_extraction(
        book: models.Book,
        agent: Any,
        deps: models.AgentDeps,
        categories: list[str],
        config: models.RunConfiguration,
        storage: StorageProvider,
        *args: Any,
    ) -> dict[int, dict[str, list[models.ExtractedEntity]]]:
        user_id = str(config.user_id)
        visits.append((storage, storage.path, user_id))
        if len(visits) == 2:
            ready.set()
        await ready.wait()
        storage.save_extraction(1, {user_id: []}, book.title)
        return {}

    with (
        patch(
            "lorebinders.workflow.extract_book",
            side_effect=concurrent_extraction,
        ),
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("lorebinders.workflow.summarize_binder", new_callable=AsyncMock),
        patch("lorebinders.workflow.generate_pdf_report"),
    ):
        outputs = await asyncio.wait_for(
            asyncio.gather(
                *(
                    build_binder(
                        run_config.model_copy(update={"user_id": user_id})
                    )
                    for user_id in ("first-user", "second-user")
                )
            ),
            timeout=5,
        )
    assert visits[0][0] is not visits[1][0]
    for storage, original_path, user_id in visits:
        assert storage.path == original_path
        assert user_id in original_path.parts
        assert storage.load_extraction(1, "Test Book") == {user_id: []}
    assert outputs[0].parent != outputs[1].parent


@pytest.mark.anyio
@pytest.mark.parametrize("dynamic", [False, True])
async def test_injected_agent_prompts_disable_cache_reuse(
    run_config: models.RunConfiguration,
    temp_workspace: Path,
    dynamic: bool,
) -> None:
    text = ["First prompt"]

    def make_agent() -> Agent[models.AgentDeps, models.ExtractionResult]:
        agent = Agent(
            TestModel(custom_output_args={"results": []}),
            deps_type=models.AgentDeps,
            output_type=models.ExtractionResult,
            system_prompt=() if dynamic else text[0],
        )
        if dynamic:

            @agent.system_prompt
            def dynamic_prompt() -> str:
                return text[0]

        return agent

    observations: list[models.ObservationEvent] = []
    with (
        patch(
            "lorebinders.workflow.analyze_entities",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("lorebinders.workflow.summarize_binder", new_callable=AsyncMock),
        patch("lorebinders.workflow.generate_pdf_report"),
    ):
        agent = make_agent()
        first = await build_binder(
            run_config, extraction_agent=agent, on_observe=observations.append
        )
        same = await build_binder(
            run_config, extraction_agent=agent, on_observe=observations.append
        )
        text[0] = "Updated prompt"
        changed = await build_binder(
            run_config,
            extraction_agent=agent if dynamic else make_agent(),
            on_observe=observations.append,
        )
    assert len({first.parent, same.parent, changed.parent}) == 3
    assert (
        sum(
            event.type == models.ObservationType.AGENT_RUN_STARTED
            for event in observations
        )
        == 3
    )
