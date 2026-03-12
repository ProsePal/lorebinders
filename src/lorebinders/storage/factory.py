"""Factory functions for storage backends.

This module exposes a singleton get_storage() for the storage provider.
"""

import functools

from lorebinders.storage.provider import StorageProvider


@functools.lru_cache(maxsize=1)
def get_storage(
    provider: type[StorageProvider],
) -> StorageProvider:
    """Get the process-wide storage provider.

    Args:
        provider: The storage provider to use.

    Returns:
        StorageProvider: A singleton storage implementation.
    """
    return provider()
