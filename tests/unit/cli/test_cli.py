from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lorebinders import models
from lorebinders.cli import cli

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(cli, ["main", "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_cli_requires_book() -> None:
    result = runner.invoke(cli, ["--author", "Test", "--series-title", "Test"])
    assert result.exit_code != 0
    assert "At least one --book must be provided" in result.output


def test_cli_requires_author() -> None:
    result = runner.invoke(
        cli, ["--book", "test.txt", "--series-title", "Test"]
    )
    assert result.exit_code != 0


def test_cli_requires_series_title() -> None:
    result = runner.invoke(cli, ["--book", "test.txt", "--author", "Test"])
    assert result.exit_code != 0


def test_cli_accepts_expected_arguments(tmp_path: Path) -> None:
    book_path = tmp_path / "book.txt"
    book_path.write_text("content")

    result = runner.invoke(
        cli,
        [
            "--book",
            str(book_path),
            "--author",
            "Jane Doe",
            "--series-title",
            "My Series",
            "--help",
        ],
    )
    assert "Usage" in result.output


def test_cli_forwards_operator_prompt_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorebinders import app

    book_path = tmp_path / "book.txt"
    book_path.write_text("Chapter 1\nAlice")
    stages = ("extraction", "analysis", "summarization", "alias_resolution")
    expected = {stage: tmp_path / f"{stage}.txt" for stage in stages}

    def run_with_prompts(
        config: models.RunConfiguration, **kwargs: Any
    ) -> Path:
        assert kwargs["prompt_files"] == expected
        return tmp_path / "report.pdf"

    monkeypatch.setattr(app, "run", run_with_prompts)
    options = [
        item
        for stage in stages
        for item in (
            f"--{stage.replace('_', '-')}-prompt",
            str(expected[stage]),
        )
    ]
    result = runner.invoke(
        cli,
        [
            "--book",
            str(book_path),
            "--author",
            "Author",
            "--series-title",
            "Series",
            *options,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Complete!" in result.output
