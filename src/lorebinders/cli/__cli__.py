import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)

from lorebinders import app, models
from lorebinders.cli.configuration import build_run_configuration
from lorebinders.logging import configure_logging

console = Console()
logger = logging.getLogger(__name__)


class ProgressHandler:
    """Rich-based progress handler for the application."""

    def __init__(self, progress: Progress) -> None:
        """Initialize the handler with a Rich Progress instance."""
        self.progress = progress
        self.tasks: dict[str, TaskID] = {}

    def __call__(self, update: models.ProgressUpdate) -> None:
        """Process a progress update."""
        if update.stage not in self.tasks:
            self.tasks[update.stage] = self.progress.add_task(
                f"[cyan]{update.stage.capitalize()}...", total=update.total
            )

        task_id = self.tasks[update.stage]
        self.progress.update(
            task_id, completed=update.current, description=update.message
        )


def _setup_logging(log_file: Path | None, verbose: bool) -> None:
    """Configure application logging."""
    configure_logging(log_file, verbose)


cli = typer.Typer(help="LoreBinders: Create a Story Bible from your book.")


@cli.command()
def main(
    series_title: Annotated[
        str, typer.Option("--series-title", help="Overall series title")
    ],
    author_name: Annotated[str, typer.Option("--author", help="Author's name")],
    books: Annotated[
        list[str] | None,
        typer.Option(
            "--book",
            help="Path to book and optional title (e.g. 'path/to/book.txt')",
        ),
    ] = None,
    narrator_name: Annotated[
        str | None,
        typer.Option(help="Name of the narrator (if using 1st person)"),
    ] = None,
    is_1st_person: Annotated[
        bool, typer.Option(help="Whether the book is written in 1st person")
    ] = False,
    tracking: Annotated[
        Literal["nested", "flat"],
        typer.Option(
            "--tracking",
            help="Appearance tracking method: 'nested' or 'flat'",
        ),
    ] = "nested",
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
    """LoreBinders: Create a Story Bible from your book.

    Raises:
        typer.Exit: If validation fails.
    """
    if not books:
        console.print(
            "[bold red]Error:[/bold red] At least one --book must be provided"
        )
        raise typer.Exit(1)

    if tracking not in ("nested", "flat"):
        console.print(
            "[bold red]Error:[/bold red] tracking must be 'nested' or 'flat'"
        )
        raise typer.Exit(1)

    config = build_run_configuration(
        books,
        series_title,
        author_name,
        narrator_name,
        is_1st_person,
        traits,
        categories,
        tracking=tracking,
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
            output = app.run(config, progress=handler)

        console.print(f"[bold green]Complete![/bold green] Report: {output}")
    except Exception as e:
        logger.exception("LoreBinders run failed")
        console.print(f"[bold red]Build Failed:[/bold red] {e}")
        raise


if __name__ == "__main__":
    cli()
