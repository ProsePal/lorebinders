# LoreBinders

LoreBinders is an AI-powered tool designed to help authors create a comprehensive
"Story Bible" from their finished manuscripts. By analyzing your book (EPUB,
PDF, etc.), LoreBinders extracts characters, locations, and other entities,
analyzes their traits, and generates a detailed PDF report.

## Features

- **Entity Extraction**: Automatically identifies characters, locations, and
  custom categories from your manuscript.
- **Deep Analysis**: Uses AI to analyze entities for traits such as
  appearance, personality, role, and relationships.
- **Customizable**: Define your own categories and traits to track what
  matters most in your story.
- **PDF Generation**: Produces a beautifully formatted "Story Bible" PDF for
  easy reference.
- **Multi-Model Support**: Leverages OpenRouter to use the best AI models for
  extraction and analysis.

## Installation

LoreBinders requires Python 3.11 or higher. We recommend using
[uv](https://github.com/astral-sh/uv) for dependency management, but `pip`
also works.

### Using `uv` (Recommended)

You don't need to install LoreBinders globally. You can run it directly from
the source directory.


### Using `pip`

```bash
pip install .
```

## CLI Usage (Optional)

LoreBinders provides a command-line interface (CLI) for generating your Story
Bible.

### Basic Usage

Run the CLI using `uv run` from the project directory:

```bash
uv run lorebinders-cli path/to/your/book.epub --author "Author Name" --title "Book Title"
```

### Advanced Usage

You can customize the  process with various options:

```bash
uv run lorebinders-cli path/to/your/book.epub \
    --author "Author Name" \
    --title "Book Title" \
    --narrator "Character Name" \
    --is-1st-person \
    --category "Magic Items" \
    --category "Factions" \
    --trait "Power Level" \
    --verbose
```

**Options:**

- `BOOK_PATH`: Path to the ebook file (epub, pdf, etc.). **(Required)**
- `--author`: The name of the author.
- `--title`: The title of the book.
- `--narrator`: Name of the narrator (useful for 1st person POV).
- `--is-1st-person`: Flag to indicate if the book is written in the 1st
  person.
- `--category`: Custom categories to extract (can be used multiple times).
- `--trait`: Custom traits to analyze for entities (can be used multiple
  times).
- `--log-file`: Path to save execution logs.
- `--verbose`: Enable verbose output for debugging.

## Configuration

LoreBinders uses environment variables for configuration. You can set these in
your shell or use a `.env` file in the working directory.

### Key Environment Variables

- `LOREBINDERS_EXTRACTION_MODEL`: The AI model used for extracting entities.
  - Default: `openrouter:bytedance/seed-1.6-flash`
- `LOREBINDERS_ANALYSIS_MODEL`: The AI model used for analyzing entity
  traits.
  - Default: `openrouter:deepseek/deepseek-v3.2`
- `LOREBINDERS_SUMMARIZATION_MODEL`: The AI model used for summarization
  tasks.
  - Default: `openrouter:bytedance/seed-1.6-flash`
- `LOREBINDERS_WORKSPACE_BASE_PATH`: Base directory for storing intermediate
  and output files.
  - Default: `work`

## Development

To contribute to LoreBinders, you'll need to set up a development environment.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/lorebinders.git
   cd lorebinders
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Install pre-commit hooks**:

   ```bash
   uvx pre-commit install
   ```

4. **Run tests**:

   ```bash
   uv run pytest
   ```

## License

This project is licensed under the Apache-2.0 License.
