from pathlib import Path

from lorebinders.cli.configuration import build_run_configuration


def test_build_run_configuration_defaults() -> None:
    config = build_run_configuration(
        books=["test.epub"],
        series_title="Series",
        author_name="Author",
        narrator_name=None,
        is_1st_person=False,
        traits=None,
        categories=None,
    )

    assert config.books[0].path == Path("test.epub")
    assert config.series_title == "Series"
    assert config.custom_traits == {}
    assert config.custom_categories == []


def test_build_run_configuration_flat_traits() -> None:
    config = build_run_configuration(
        books=["test.epub:Title"],
        series_title="Series",
        author_name="Author",
        narrator_name=None,
        is_1st_person=False,
        traits=["Trait1", "Trait2"],
        categories=None,
    )

    assert config.books[0].title == "Title"
    assert config.custom_traits == {"Characters": ["Trait1", "Trait2"]}


def test_build_run_configuration_namespaced_traits() -> None:
    config = build_run_configuration(
        books=["test.epub"],
        series_title="Series",
        author_name="Author",
        narrator_name=None,
        is_1st_person=False,
        traits=["Trait1", "Locations:Atmosphere", "Beasts:Ferocity"],
        categories=["Beasts"],
    )

    assert "Characters" in config.custom_traits
    assert "Trait1" in config.custom_traits["Characters"]

    assert "Locations" in config.custom_traits
    assert "Atmosphere" in config.custom_traits["Locations"]

    assert "Beasts" in config.custom_traits
    assert "Ferocity" in config.custom_traits["Beasts"]

    assert "Beasts" in config.custom_categories
