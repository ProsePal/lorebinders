import subprocess
import sys


def test_lazy_import_without_sqlalchemy() -> None:
    """lorebinders.storage can be imported even if sqlalchemy is missing."""
    code = (
        "import sys\n"
        "sys.modules['sqlalchemy'] = None\n"
        "import lorebinders.storage\n"
        "print('Success')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "Success" in result.stdout
