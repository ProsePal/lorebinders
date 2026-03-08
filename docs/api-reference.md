# API Reference

This document provides a high-level reference for the core modules and classes within the LoreBinders engine.

## 📦 `lorebinders.app`

The primary interface for interacting with the LoreBinders engine.

### `build_binder(book_path, author, title, progress_callback=None)`
Starts the Story Bible generation pipeline.
- **Parameters**:
  - `book_path` (str): Path to the manuscript (EPUB, PDF, etc.).
  - `author` (str): The name of the author.
  - `title` (str): The title of the book.
  - `progress_callback` (Callable, optional): A hook for real-time progress updates.
- **Returns**: `Binder` object representing the completed Story Bible.

## 🗂 `lorebinders.models`

The fundamental data structures used throughout the engine.

### `Binder`
The root state object.
- **Attributes**:
  - `author` (str): The book's author.
  - `title` (str): The book's title.
  - `categories` (Dict[str, CategoryRecord]): Map of entity categories.

### `EntityRecord`
Represents a unique entity across the entire manuscript.
- **Attributes**:
  - `name` (str): The normalized name of the entity.
  - `appearances` (Dict[int, EntityAppearance]): Map of chapter numbers to appearances.
  - `summary` (Optional[str]): The synthesized narrative summary.

### `EntityAppearance`
Details of an entity's presence in a specific chapter.
- **Attributes**:
  - `chapter_id` (int): The chapter number.
  - `traits` (Dict[str, str]): Key-value pairs of extracted traits.
  - `evidence` (Dict[str, str]): Text snippets supporting the traits.

## 🚀 `lorebinders.workflow`

The core pipeline orchestration logic.

### `WorkflowManager`
Manages the state and transition between pipeline stages.
- **Key Methods**:
  - `run()`: Executes the full pipeline sequentially.
  - `ingest()`: Parses the manuscript into chapters.
  - `extract()`: Performs entity discovery across chapters.
  - `analyze()`: Executes deep-dive analysis of entity appearances.
  - `summarize()`: Collates entity traits into final summaries.

## 🤖 `lorebinders.agent.factory`

### `AgentFactory`
Creates specialized Pydantic-AI agents.
- **Methods**:
  - `get_extraction_agent()`: Returns an instance of the extraction agent.
  - `get_analysis_agent()`: Returns an instance of the analysis agent.
  - `get_summarization_agent()`: Returns an instance of the summarization agent.

## 💾 `lorebinders.storage.provider`

### `StorageProvider` (Protocol)
Defines the interface for all persistence backends.
- **Required Methods**:
  - `save_binder(binder)`
  - `load_binder(author, title)`
  - `save_chapter(chapter)`
  - `load_chapter(author, title, chapter_id)`
