from pathlib import Path

import mypy.api
import pytest

from numpydantic import ndarray
from numpydantic.meta import update_ndarray_stub

MYPY_DIR = Path(__file__).parent / "data" / "mypy"


@pytest.fixture(scope="session", autouse=True)
def refresh_stubs():
    """Ensure no stale stubs"""
    # ensure all interfaces are imported
    from numpydantic import interface  # noqa: F401

    stub_file = Path(ndarray.__file__).with_suffix(".pyi")
    backup_file = stub_file.with_suffix(".pyi.bak")
    if stub_file.exists():
        stub_file.rename(backup_file)

    update_ndarray_stub()
    yield
    if backup_file.exists():
        stub_file.unlink(missing_ok=True)
        backup_file.rename(stub_file)


@pytest.mark.parametrize("test_file", MYPY_DIR.glob("*.py"))
def test_mypy(test_file: Path):
    """The mypy examples should pass static type checking"""
    res = mypy.api.run([str(test_file)])
    assert res == ("Success: no issues found in 1 source file\n", "", 0)
