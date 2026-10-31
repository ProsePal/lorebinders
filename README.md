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

Installing with `pip` does not apply the transitive dependency security
constraints in `[tool.uv]`. The protections provided by those constraints
are unavailable with this installation method, so vulnerable transitive
dependencies may be installed.

For `pip` installs, use a fresh virtual environment to minimize the risk
of reusing older, vulnerable dependencies from an existing environment.
This isolation does not enforce the uv security constraints.

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

### Runtime prompt files

Trusted operators can override any stage without replacing bundled assets:

```bash
uv run --extra cli lorebinders-cli --book manuscript.epub \
    --author "Author Name" --series-title "Series Title" \
    --extraction-prompt prompts/extraction.txt \
    --analysis-prompt prompts/analysis.txt
```

The other options are `--summarization-prompt` and
`--alias-resolution-prompt`. Unspecified stages keep their bundled prompts.
Every configured file must be readable UTF-8, contain non-whitespace text,
and fit the 16,384-byte limit per stage. Overrides are read with a bounded
16,385-byte read and rejected before decoding if oversized. This input-size
budget also applies to bundled prompts and is not a model context guarantee;
all files are validated and snapshotted before any model call, even when a
stage is disabled. These paths are trusted local operator inputs and should
not be accepted directly from public API clients.

Python callers can pass `prompt_files={"extraction": Path("prompt.txt")}`
to `lorebinders.run` or `lorebinders.workflow.build_binder`. Accepted keys are
`extraction`, `analysis`, `summarization`, and `alias_resolution`.
`lorebinders.prompts.build_prompt_loader(prompt_files)` exposes the same
snapshot loader for direct agent runs. It accepts filenames such as
`extraction.txt`; hash its returned text encoded as UTF-8 to record the
effective prompt identity for reproducibility.

Prompt files and manuscript inputs must remain immutable while a build is
starting. For coordinated updates, publish a new versioned directory and
point the next build at it; the loader does not provide atomic multi-file
publication. After loading, each effective prompt remains fixed for that run.
Operators are responsible for fitting prompts plus manuscript text and
requested output within the selected model's context budget.

All builds use an internal cache workspace suffix derived from effective
prompts, source bytes, narrator/run configuration, settings, and model names.
Identical inputs with engine-created agents reuse cached extractions,
profiles, and summaries; changed inputs use a separate workspace. Injecting
any caller-supplied agent disables cache reuse for the entire build, including
agents with static or dynamic prompts, because their complete behavior is
not represented by the bundled prompt loader. Each build constructs its own
provider instance, so concurrent builds do not share mutable workspace state.
Historical workspaces remain intact but are
not reused by this cache-identity version, including for bundled defaults.
The user/run prefix, human series title, and report filename remain unchanged.
Custom storage providers must implement
`set_workspace(author, title, user_id=None, *, cache_namespace=None)` and
scope every extraction, profile, and summary cache key by that namespace.
Ignoring it can reuse stale results. The engine passes the original title
and namespace separately so providers can preserve their user/run prefix.
Built-in filesystem and database providers support this contract. Known
`httpx.Timeout` and path settings have stable identities; opaque or dynamic
setting values conservatively disable cache reuse for that build.
Existing entity normalization and deduplication behavior is preserved.
The extraction instructions accept names longer than two words within the
existing 200-character limit. Leading titles can still be stripped and
contained location names can still be merged downstream; these changes do
not promise distinct final entities for those cases.

Operators can remove one local cache workspace with
`clean_workspace(author, title, user_id=user_id, cache_namespace=namespace)`
from `lorebinders.storage.workspace`. Omitting the namespace cleans only the
legacy unsuffixed workspace. Namespace creation and cleanup share validation
and title suffix rules; other namespace directories remain intact.

## Configuration

LoreBinders uses environment variables for configuration. You can set these in
your shell or use a `.env` file in the working directory.

### Key Environment Variables

- `LOREBINDERS_SPEND_CEILING`: Optional soft USD spend ceiling per pipeline
  run. Disabled by default. In-flight requests and estimated prices can cause
  the final cost to exceed the ceiling.
- `LOREBINDERS_MAX_OUTPUT_TOKENS`: Optional maximum generated tokens per
  model response. Unset by default, leaving the provider's limit in effect.
  Reasoning tokens count toward this limit where the provider uses a shared
  completion budget.
- `LOREBINDERS_EXTRACTION_MAX_OUTPUT_TOKENS`: Optional extraction-specific
  response limit. Defaults to the general limit when that is configured.
  Per-call settings may lower configured limits but cannot raise them.
- `LOREBINDERS_EXTRACTION_MODEL`: The AI model used for extracting entities.
  - Default: `openrouter:bytedance/seed-1.6-flash`
- `LOREBINDERS_ANALYSIS_MODEL`: The AI model used for analyzing entity
  traits.
  - Default: `openrouter:deepseek/deepseek-v3.2`
- `LOREBINDERS_SUMMARIZATION_MODEL`: The AI model used for summarization
  tasks.
  - Default: `openrouter:bytedance/seed-1.6-flash`
- `LOREBINDERS_FAILURE_THRESHOLD`: Maximum tolerable failure ratio (0.0 to 1.0)
  for pipeline stages (extraction, analysis, summarization) before an error
  is raised.
  - Default: `0.2`
- `LOREBINDERS_FAILURE_THRESHOLD_MIN_COUNT`: Minimum number of failed tasks
  required in a pipeline stage before `failure_threshold` aborts execution.
  Prevents aborting small books or small entity sets on isolated failures.
  - Default: `2`
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
