# Codebase Summary

This document provides a detailed overview of the LoreBinders project's structure, modules, and their responsibilities.

## 📁 Directory Structure

```text
src/lorebinders/
├── __init__.py           # Package entry point.
├── app.py                # Main engine orchestration and high-level logic.
├── logging.py            # Custom logging setup.
├── models.py             # Pydantic models for data hierarchy (Binder, EntityRecord, etc.).
├── settings.py           # Configuration management via Pydantic-Settings.
├── types.py              # Common type definitions.
├── workflow.py           # Core pipeline workflow implementation.
├── agent/                # AI Agent implementations.
│   ├── analysis.py       # Deep-dive entity analysis agent.
│   ├── extraction.py     # Entity extraction agent.
│   ├── factory.py        # Agent creation factory using Pydantic-AI.
│   ├── settings.py       # Agent-specific configuration.
│   └── summarization.py  # Narrative summarization agent.
├── cli/                  # Command-Line Interface.
│   ├── __cli__.py        # Typer/Rich CLI implementation.
│   └── configuration.py  # CLI-specific configuration and validation.
├── refinement/           # Post-processing and data cleaning logic.
│   ├── cleaning.py       # Data sanitization and hallunication removal.
│   ├── conversion.py     # Data format conversions.
│   ├── deduplication.py  # Alias merging and entity resolution.
│   ├── normalization.py  # Name and trait normalization.
│   ├── patterns.py       # Regex patterns for text processing.
│   └── sorting.py        # Entity and category sorting logic.
├── reporting/            # PDF report generation.
│   ├── pdf.py            # ReportLab PDF generator.
│   └── styles.py         # Stylesheet definitions for the PDF.
└── storage/              # Data persistence layer.
    ├── factory.py        # Storage provider factory.
    ├── provider.py       # StorageProvider Protocol definition.
    ├── workspace.py      # Local workspace and directory management.
    └── providers/        # Concrete storage backends.
        ├── db.py         # SQLAlchemy relational database provider.
        ├── file.py       # Filesystem/JSON provider.
        └── test.py       # Mock/In-memory provider for testing.
```

## 🏗 Key Components

### Core Engine (`app.py`, `workflow.py`)
Responsible for managing the sequential pipeline: Ingestion -> Extraction -> Refinement -> Analysis -> Aggregation -> Summarization -> Delivery.

### Data Models (`models.py`)
The "source of truth" for all structured data. LoreBinders uses strictly typed hierarchies to ensure data integrity throughout the transformation pipeline.

### Agents (`agent/`)
Pydantic-AI powered agents that interact with LLMs. Each agent has a specific scope (extraction, analysis, or summarization) and utilizes structured outputs to maintain data consistency.

### Refinement Engine (`refinement/`)
A collection of logic to ensure the data produced by LLMs is normalized, deduplicated, and free of common AI artifacts.

### Storage Abstraction (`storage/`)
A protocol-based system that allows the engine to persist data to either local files or a relational database without changing core logic.

### Reporting Layer (`reporting/`)
Converts the final `Binder` state into a professional PDF Story Bible.
