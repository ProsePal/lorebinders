"""Storage package for persistence and workspace management."""

from typing import Any

from lorebinders.storage.factory import get_storage
from lorebinders.storage.provider import StorageProvider
from lorebinders.storage.providers.file import FilesystemStorage
from lorebinders.storage.workspace import sanitize_filename

__all__ = [
    "get_storage",
    "StorageProvider",
    "FilesystemStorage",
    "DBStorage",
    "sanitize_filename",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazily import optional dependencies."""
    if name == "DBStorage":
        from lorebinders.storage.providers.db import DBStorage

        return DBStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
