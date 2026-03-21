from pathlib import Path

from typer.testing import CliRunner

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
