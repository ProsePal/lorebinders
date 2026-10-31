from pathlib import Path

import pytest

from lorebinders.prompts import build_prompt_loader


def test_prompt_overrides_are_snapshotted_with_bundled_fallback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("Custom extraction\n", encoding="utf-8")
    loader = build_prompt_loader({"extraction": path})
    path.write_text("Changed", encoding="utf-8")
    assert loader("extraction.txt") == "Custom extraction\n"
    assert loader("analysis.txt") == build_prompt_loader()("analysis.txt")
    assert loader("alias_resolution.txt")
    assert loader("summarization.txt")


@pytest.mark.parametrize(
    "stage", ["extraction", "analysis", "summarization", "alias_resolution"]
)
def test_all_stages_support_overrides(tmp_path: Path, stage: str) -> None:
    path = tmp_path / "prompt.txt"
    path.write_text("Custom stage prompt", encoding="utf-8")
    assert (
        build_prompt_loader({stage: path})(f"{stage}.txt")
        == "Custom stage prompt"
    )


def test_missing_override_fails_eagerly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_prompt_loader({"analysis": tmp_path / "missing"})


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_override_fails_eagerly(tmp_path: Path, content: str) -> None:
    path = tmp_path / "empty"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        build_prompt_loader({"summarization": path})


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown prompt stage"):
        build_prompt_loader({"typo": tmp_path / "missing"})


def test_loader_rejects_arbitrary_paths() -> None:
    with pytest.raises(ValueError, match="Unknown prompt"):
        build_prompt_loader()("../secret.txt")


def test_prompt_byte_budget_accepts_boundary_and_rejects_oversize(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompt.txt"
    path.write_bytes("é".encode() * 8192)
    assert (
        build_prompt_loader({"extraction": path})("extraction.txt")
        == "é" * 8192
    )
    path.write_bytes("é".encode() * 8193)
    with pytest.raises(ValueError, match="16384"):
        build_prompt_loader({"extraction": path})
