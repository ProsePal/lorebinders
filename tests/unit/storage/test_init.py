import pytest

import lorebinders.storage as storage


def test_getattr_dbstorage() -> None:
    db_storage = storage.DBStorage
    assert db_storage.__name__ == "DBStorage"


def test_getattr_missing() -> None:
    with pytest.raises(
        AttributeError,
        match="module 'lorebinders.storage' has no attribute 'MissingAttr'",
    ):
        _ = storage.MissingAttr
