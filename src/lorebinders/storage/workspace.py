import re
import shutil
from pathlib import Path

from lorebinders.settings import get_settings


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename.

    Replaces non-alphanumeric characters (except -) with underscores.

    Args:
        name: The input string.

    Returns:
        A safe filename string.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\-]", "_", name)
    return re.sub(r"_+", "_", cleaned)


def ensure_workspace(
    author: str,
    title: str,
    base_path: Path | None = None,
    user_id: str | None = None,
) -> Path:
    """Create (if needed) and return the path to the book's workspace.

    Structure: {base_path}/{user_id}/{author}/{title} or
    {base_path}/{author}/{title}

    Args:
        author: The name of the author.
        title: The title of the book.
        base_path: Root directory for workspaces. Defaults to "work".
        user_id: Optional user ID for namespacing.

    Returns:
        The Path to the book's workspace directory.
    """
    base = (
        base_path
        if base_path is not None
        else get_settings().workspace_base_path
    )
    safe_author = sanitize_filename(author)
    safe_title = sanitize_filename(title)

    if user_id:
        safe_user_id = sanitize_filename(user_id)
        path = base / safe_user_id / safe_author / safe_title
    else:
        path = base / safe_author / safe_title
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_workspace(
    author: str,
    title: str,
    base_path: Path | None = None,
    user_id: str | None = None,
) -> None:
    """Remove the workspace directory for a specific book.

    Args:
        author: The name of the author.
        title: The title of the book.
        base_path: Root directory for workspaces. Defaults to "work".
        user_id: Optional user ID for namespacing.

    Raises:
        ValueError: If the resolved path escapes the workspace boundary.
    """
    base = (
        base_path
        if base_path is not None
        else get_settings().workspace_base_path
    )
    safe_author = sanitize_filename(author)
    safe_title = sanitize_filename(title)

    if user_id:
        safe_user_id = sanitize_filename(user_id)
        path = base / safe_user_id / safe_author / safe_title
    else:
        path = base / safe_author / safe_title

    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Path {path} escapes workspace boundary")

    if path.exists():
        shutil.rmtree(path)
