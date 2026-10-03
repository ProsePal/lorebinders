from pathlib import Path

from pypdf import PdfReader

from lorebinders.models import Binder
from lorebinders.reporting.pdf import _esc, generate_pdf_report


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


def test_generate_pdf_report_with_stage_failures(tmp_path: Path) -> None:
    output_path = tmp_path / "test_report_failures.pdf"

    binder = Binder()
    binder.stage_failures = {
        "extraction": {"failed_count": 2, "total_count": 10},
        "analysis": {"failed_count": 5, "total_count": 50},
    }

    generate_pdf_report(binder, output_path)

    assert output_path.exists()

    reader = PdfReader(output_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    assert "Note: Partial Results" in text
    assert "Extraction stage: 2/10 chapters failed." in text
    assert "Analysis stage: 5/50 chapter blocks failed." in text


MARKUP_PAYLOADS = [
    '<img src="file:///etc/passwd"/>',
    '<onDraw name="exploit"/>',
    "<b>bold injection</b>",
    "<i>italic injection</i>",
    "Tom & Jerry",
    'He said "hello"',
    "a < b > c",
    "<<double>>",
    "</Paragraph>",
]


def test_esc_escapes_xml_special_characters() -> None:
    """_esc must convert <, >, and & to XML entities.

    xml.sax.saxutils.escape does not escape double-quotes by default;
    quotes are only significant inside attribute values, which Paragraph
    markup never uses, so the default behaviour is correct here.
    """
    assert _esc("<") == "&lt;"
    assert _esc(">") == "&gt;"
    assert _esc("&") == "&amp;"
    assert "&lt;" in _esc('<img src="file:///etc/passwd"/>')
    assert "&amp;" in _esc("Tom & Jerry")


def test_esc_roundtrips_plain_text() -> None:
    """_esc must leave strings with no special characters unchanged."""
    assert _esc("Hero") == "Hero"
    assert _esc("Book 1, Chapter 3") == "Book 1, Chapter 3"
    assert _esc("The hero is strong.") == "The hero is strong."


def test_esc_coerces_non_strings() -> None:
    """_esc must accept non-string input by coercing via str()."""
    assert _esc(42) == "42"
    assert _esc(3.14) == "3.14"


def test_pdf_markup_injection_entity_name(tmp_path: Path) -> None:
    """Entity names with markup payloads must render as literal text.

    Generation must succeed and representative tag-text must appear
    verbatim in the PDF text layer, not be interpreted or silently dropped.
    """
    output_path = tmp_path / "inject_name.pdf"
    binder = Binder()
    for payload in MARKUP_PAYLOADS:
        binder.add_appearance(
            "Characters",
            payload,
            1,
            "Book",
            {"trait": "value"},
        )

    generate_pdf_report(binder, output_path)
    assert output_path.exists()

    reader = PdfReader(output_path)
    text = "".join(page.extract_text() for page in reader.pages)

    assert "Tom & Jerry" in text
    assert "a < b > c" in text
    assert "img" in text
    assert "onDraw" in text
    assert "bold injection" in text


def test_pdf_markup_injection_summary(tmp_path: Path) -> None:
    """Entity summaries with markup payloads must render as literal text.

    Generation must succeed and tag-text from the payload must appear
    verbatim in the PDF text layer, not be interpreted or silently dropped.
    """
    output_path = tmp_path / "inject_summary.pdf"
    binder = Binder()
    binder.add_appearance("Characters", "Hero", 1, "Book", {"trait": "value"})
    binder.categories["Characters"].entities[
        "Hero"
    ].summary = '<img src="file:///etc/passwd"/> The hero is strong.'

    generate_pdf_report(binder, output_path)
    assert output_path.exists()

    reader = PdfReader(output_path)
    text = "".join(page.extract_text() for page in reader.pages)
    assert "The hero is strong." in text
    assert "img" in text


def test_pdf_markup_injection_trait_name(tmp_path: Path) -> None:
    """Trait names with markup payloads must render as literal text.

    Generation must succeed and representative tag-text must appear
    verbatim in the PDF text layer, not be interpreted or silently dropped.
    """
    output_path = tmp_path / "inject_trait.pdf"
    binder = Binder()
    for i, payload in enumerate(MARKUP_PAYLOADS, start=1):
        binder.add_appearance(
            "Characters",
            "Hero",
            i,
            "Book",
            {payload: "some value"},
        )

    generate_pdf_report(binder, output_path)
    assert output_path.exists()

    reader = PdfReader(output_path)
    text = "".join(page.extract_text() for page in reader.pages)

    assert "Tom & Jerry" in text
    assert "img" in text
    assert "onDraw" in text


def test_pdf_markup_injection_trait_value(tmp_path: Path) -> None:
    """Trait values with markup payloads must render as literal text.

    Generation must succeed and representative tag-text must appear
    verbatim in the PDF text layer, not be interpreted or silently dropped.
    """
    output_path = tmp_path / "inject_value.pdf"
    binder = Binder()
    for i, payload in enumerate(MARKUP_PAYLOADS, start=1):
        binder.add_appearance(
            "Characters",
            "Hero",
            i,
            "Book",
            {"Physique": payload},
        )

    generate_pdf_report(binder, output_path)
    assert output_path.exists()

    reader = PdfReader(output_path)
    text = "".join(page.extract_text() for page in reader.pages)

    assert "Tom & Jerry" in text
    assert "img" in text
    assert "onDraw" in text


def test_pdf_markup_injection_category_name(tmp_path: Path) -> None:
    """Category names with markup payloads must render as literal text.

    Generation must succeed and representative tag-text must appear
    verbatim in the PDF text layer, not be interpreted or silently dropped.
    """
    output_path = tmp_path / "inject_category.pdf"
    binder = Binder()
    for payload in MARKUP_PAYLOADS:
        binder.add_appearance(
            payload,
            "Entity",
            1,
            "Book",
            {"trait": "value"},
        )

    generate_pdf_report(binder, output_path)
    assert output_path.exists()

    reader = PdfReader(output_path)
    text = "".join(page.extract_text() for page in reader.pages)

    assert "Tom & Jerry" in text
    assert "img" in text
    assert "onDraw" in text
