import json
import shutil
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from typer.testing import CliRunner

from lorebinders import app
from lorebinders.agent import (
    create_analysis_agent,
    create_extraction_agent,
    create_summarization_agent,
)
from lorebinders.cli.configuration import build_run_configuration
from lorebinders.models import (
    AnalysisResult,
    AnalyzedTrait,
    CategoryEntities,
    ExtractedEntity,
    ExtractionResult,
    SummarizerResult,
)

runner = CliRunner()


def test_e2e_ingestion_flow(
    tmp_path: Path,
) -> None:
    book_content = (
        "Project Genesis\n"
        "***\n"
        "Chapter 1\n"
        "It was a dark and stormy night.\n"
        "***\n"
        "Chapter 2\n"
        "The sun came out."
    )

    source_file = tmp_path / "genesis.txt"
    source_file.write_text(book_content)

    cleanup_target = Path.cwd() / "work" / "Test_Author"

    config = build_run_configuration(
        [f"{str(source_file)}:Project Genesis"],
        series_title="Project Genesis",
        author_name="Test Author",
        narrator_name=None,
        is_1st_person=False,
        traits=None,
        categories=None,
    )

    def mock_extract(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=ExtractionResult(
                        results=[
                            CategoryEntities(
                                category="Locations",
                                entities=[
                                    ExtractedEntity(
                                        name="Night",
                                        presence_type="literal_entity",
                                    )
                                ],
                            )
                        ]
                    ).model_dump_json()
                )
            ]
        )

    def mock_analyze(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        results = [
            AnalysisResult(
                entity_name="Night",
                category="Locations",
                traits=[
                    AnalyzedTrait(
                        trait="Key Features", value="Dark", evidence="..."
                    )
                ],
            )
        ]

        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {"response": [i.model_dump() for i in results]}
                    )
                )
            ]
        )

    def mock_summarize(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    content=SummarizerResult(
                        entity_name="Night", summary="Summary"
                    ).model_dump_json()
                )
            ]
        )

    ext_agent = create_extraction_agent()
    ana_agent = create_analysis_agent()
    sum_agent = create_summarization_agent()

    with (
        ext_agent.override(model=FunctionModel(mock_extract)),
        ana_agent.override(model=FunctionModel(mock_analyze)),
        sum_agent.override(model=FunctionModel(mock_summarize)),
    ):
        try:
            output_path = app.run(
                config,
                extraction_agent=ext_agent,
                analysis_agent=ana_agent,
                summarization_agent=sum_agent,
            )

            assert output_path.exists()

        finally:
            if cleanup_target.exists():
                shutil.rmtree(cleanup_target)
