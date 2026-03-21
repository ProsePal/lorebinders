from pathlib import Path

from pypdf import PdfReader

from lorebinders.models import Binder
from lorebinders.reporting.pdf import generate_pdf_report


def test_generate_pdf_report_aggregated(tmp_path: Path) -> None:
    output_path = tmp_path / "test_report.pdf"

    binder = Binder()
    binder.add_appearance(
        "Characters",
        "Hero",
        1,
        "Book 1",
        {"Physique": "Lean", "Personality": "Brave"},
    )
    binder.add_appearance(
        "Characters",
        "Hero",
        2,
        "Book 1",
        {"Physique": "Muscular", "Personality": "Brave"},
    )
    binder.categories["Characters"].entities[
        "Hero"
    ].summary = "The hero is strong."

    binder.add_appearance(
        "Settings", "Castle", 1, "Book 1", {"Atmosphere": "Dark"}
    )

    generate_pdf_report(binder, output_path)

    assert output_path.exists()

    reader = PdfReader(output_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    assert "LoreBinders Story Bible" in text
    assert "Characters" in text
    assert "Settings" in text

    assert "Hero" in text
    assert "Castle" in text

    assert "The hero is strong." in text

    assert "Physique" in text
    assert "Personality" in text
    assert "Atmosphere" in text

    assert "Book 1, Chapter 1: Lean" in text
    assert "Book 1, Chapter 2: Muscular" in text
    assert "Book 1, Chapter 1: Dark" in text

    assert "Book 1, Chapter 1: Brave" in text
    assert "Book 1, Chapter 2: Brave" in text
