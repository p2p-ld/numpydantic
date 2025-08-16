from pathlib import Path

import mypy.api
import pytest

MYPY_DIR = Path(__file__).parent / "data" / "mypy"


@pytest.mark.parametrize("test_file", MYPY_DIR.glob("*.py"))
def test_mypy(test_file: Path):
    """The mypy examples should pass static type checking"""
    res = mypy.api.run([str(test_file)])
    assert res == ("Success: no issues found in 1 source file\n", "", 0)
