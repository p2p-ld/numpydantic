from pathlib import Path

import mypy.api
import pytest

from numpydantic import ndarray, update_ndarray_stub
from numpydantic.testing.cases import MYPY_TYPING_CASES
from numpydantic.testing.surfaces import AnnotationForm

DATA_DIR = Path(__file__).parent / "data"
MYPY_DIR = DATA_DIR / "mypy"

pytestmark = pytest.mark.typechecking



@pytest.fixture(scope="session")
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


@pytest.fixture
def blank_config(tmp_path) -> Path:
    """Nothing to see here, just a blank file without the plugins configured"""
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text("[tool.mypy]")
    return config_path


@pytest.mark.parametrize("test_file", (DATA_DIR / "stub" / "correct").glob("*.py"))
def test_mypy_noplugin(test_file: Path, refresh_stubs, blank_config):
    """The mypy examples should pass static type checking with only the stub"""
    res = mypy.api.run(["--config-file", str(blank_config), str(test_file)])
    assert "Success: no issues found in 1 source file" in res[0]
    assert res[1] == ""
    assert res[2] == 0
