import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from lorebinders import app
from lorebinders.cli.configuration import build_run_configuration
from lorebinders.logging import configure_logging
from lorebinders.models import ProgressUpdate

cli = typer.Typer(no_args_is_help=True)
console = Console()
logger = logging.getLogger("lorebinders.cli")


class ProgressHandler:
    """Handles rich progress updates from the application."""

    def __init__(self, progress: Progress) -> None:
        """Initialise tasks for each pipeline stage."""
        self.progress = progress
        self.extraction_task = progress.add_task("Extracting...", total=None)
        self.analysis_task = progress.add_task(
            "Analyzing...", total=None, visible=False
        )
        self.summarization_task = progress.add_task(
            "Summarizing...", total=None, visible=False
        )

    def __call__(self, update: ProgressUpdate) -> None:
        """Route a progress update to the correct rich task."""
        match update.stage:
            case "extraction":
                self.progress.update(
                    self.extraction_task,
                    completed=update.current,
                    total=update.total,
                    description=update.message,
                )
            case "analysis":
                self.progress.update(
                    self.analysis_task,
                    visible=True,
                    completed=update.current,
                    total=update.total,
                    description=update.message,
                )
            case "summarization":
                self.progress.update(
                    self.summarization_task,
                    visible=True,
                    completed=update.current,
                    total=update.total,
                    description=update.message,
                )


def _setup_logging(log_file: Path | None, verbose: bool) -> None:
    if log_file or verbose:
        configure_logging(log_file)
    if verbose:
        logging.getLogger("lorebinders").setLevel(logging.DEBUG)


@cli.command()
def main(
    book_path: Annotated[
        Path,
        typer.Argument(
            exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ],
    author_name: Annotated[str, typer.Option("--author", help="Author's name")],
    book_title: Annotated[str, typer.Option("--title", help="Book title")],
    narrator_name: Annotated[
        str | None,
        typer.Option(help="Name of the narrator (if using 1st person)"),
    ] = None,
    is_1st_person: Annotated[
        bool, typer.Option(help="Whether the book is written in 1st person")
    ] = False,
    traits: Annotated[
        list[str] | None, typer.Option("--trait", help="Custom trait to track")
    ] = None,
    categories: Annotated[
        list[str] | None,
        typer.Option("--category", help="Custom category to track"),
    ] = None,
    log_file: Annotated[
        Path | None, typer.Option("--log-file", help="Path to save logs")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Enable verbose logging")
    ] = False,
) -> None:
    """LoreBinders: Create a Story Bible from your book."""
    config = build_run_configuration(
        book_path,
        author_name,
        book_title,
        narrator_name,
        is_1st_person,
        traits,
        categories,
    )
    _setup_logging(log_file, verbose)

    console.print("[bold blue]Starting LoreBinders...[/bold blue]")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            handler = ProgressHandler(progress)
            output = app.run(config, progress=handler, log_file=log_file)

        console.print(f"[bold green]Complete![/bold green] Report: {output}")
    except Exception as e:
        logger.exception("LoreBinders run failed")
        console.print(f"[bold red]Build Failed:[/bold red] {e}")
        raise


if __name__ == "__main__":
    cli()
