# Code Standards and Conventions

LoreBinders follows strict coding standards to ensure code quality, maintainability, and consistency. All contributions must adhere to these guidelines.

## 🛠 Tooling

We use the following tools for linting, formatting, and type checking:

- **Ruff**: Fast Python linter and formatter.
- **MyPy**: Static type checker for Python.
- **Pre-commit**: Automated checks before every commit.
- **Pytest**: For unit, integration, and end-to-end testing.

## 🐍 Python Guidelines

- **Version**: Python 3.10+
- **Type Annotations**: Mandatory for all functions and classes.
- **Docstrings**: We use the **Google** convention for docstrings.
- **Naming Conventions**:
  - `Snake_case` for variables, functions, and modules.
  - `PascalCase` for classes.
  - `SCREAMING_SNAKE_CASE` for constants.
- **Import Ordering**: Handled by Ruff (`isort`).
- **Line Length**: Maximum 80 characters.

## 🔍 Linting Rules (Ruff)

We enable a broad set of Ruff rules, including:
- `A`: Built-ins (prevent shadowing)
- `ANN`: Annotations (ensure type hints)
- `D`: Pydocstyle (enforce docstring standards)
- `DOC`: Docstring formatting
- `E`, `F`: Pycodestyle and Pyflakes
- `I`: Isort
- `N`: PEP8-naming
- `UP`: Pyupgrade (modern Python syntax)

## ✅ Type Checking (MyPy)

- All code must be typed.
- We target strict typing where possible.
- MyPy configuration is located in `pyproject.toml`.

## 🧪 Testing Standards

- **Coverage Requirement**: Minimum **80%** code coverage.
- **Test Structure**:
  - `tests/unit`: Isolated tests for individual functions and classes.
  - `tests/integration`: Tests for interactions between components (e.g., agents).
  - `tests/e2e`: Full system tests including CLI usage.
- **Mocking**: Use standard `unittest.mock` or `pytest-mock` to avoid external API calls during unit tests.

## 📦 Dependency Management

- Use **`uv`** for managing dependencies and virtual environments.
- Add new dependencies via `uv add`.
- Keep `pyproject.toml` updated.

## 🔄 Commit Workflow

Before committing code:
1. Run `ruff check --fix .` and `ruff format .`.
2. Run `mypy .`.
3. Run `pytest`.
4. Ensure pre-commit hooks pass.
