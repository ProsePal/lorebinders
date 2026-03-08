# Deployment and Configuration Guide

This guide provides instructions for installing, configuring, and deploying LoreBinders.

## 📦 Installation

LoreBinders requires **Python 3.10+** and is managed using the `uv` tool.

### Local Development
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `uv sync`
3. Install pre-commit hooks: `pre-commit install`

### Production Setup
1. Ensure `uv` is installed: `pip install uv`
2. Install the package: `uv pip install .`

## ⚙️ Configuration

LoreBinders uses environment variables for configuration. You can set these in your shell or a `.env` file in the root directory.

### Core Settings
- `LOREBINDERS_EXTRACTION_MODEL`: AI model for entity extraction (Default: `openrouter:bytedance/seed-1.6-flash`).
- `LOREBINDERS_ANALYSIS_MODEL`: AI model for trait analysis (Default: `openrouter:deepseek/deepseek-v3.2`).
- `LOREBINDERS_SUMMARIZATION_MODEL`: AI model for synthesis (Default: `openrouter:bytedance/seed-1.6-flash`).
- `LOREBINDERS_WORKSPACE_BASE_PATH`: Base path for intermediate processing and outputs (Default: `work`).

### LLM API Access
LoreBinders currently leverages **OpenRouter**. Ensure your `OPENROUTER_API_KEY` is set in your environment or `.env` file.

## 🚀 Execution

### Using the CLI
Run the CLI using `uv run`:
```bash
uv run lorebinders-cli <book_path> --author "<author_name>" --title "<book_title>"
```

### Integrated Usage
LoreBinders can be used programmatically in any Python application:
```python
from lorebinders.app import build_binder

binder = build_binder(
    book_path="path/to/manuscript.epub",
    author="John Doe",
    title="My Great Story"
)
print(f"Binder for {binder.title} by {binder.author} generated.")
```

## 🛠 Logging

Logs are stored by default in the `logs/` directory. You can specify a custom log file path using the `--log-file` flag in the CLI.

## 💾 Persistence

The default storage is file-based (`work/`). To use a relational database, ensure `sqlalchemy` is installed and configure the storage provider in `settings.py`.
