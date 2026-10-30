import subprocess
from pathlib import Path

import pyright
import pytest

from tests.fixtures.typecheck import check_errors, shared_params

pytestmark = [pytest.mark.typechecking, pytest.mark.pyright]


@pytest.mark.parametrize("test_file", shared_params("correct", "pyright"))
def test_pyright_correct(test_file: Path) -> None:
    """The mypy examples should pass static type checking with pyright too."""
    res = pyright.run(
        str(test_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    err = res.stderr.decode("utf-8")
    out = res.stdout.decode("utf-8")
    assert res.returncode == 0, f"stderr:\n{err}\n\nstdout:\n{out}"


@pytest.mark.parametrize("test_file", shared_params("incorrect", "pyright"))
def test_pyright_incorrect(test_file: Path) -> None:
    """Pyright refuses the lines we expect it to"""
    res = pyright.run(
        str(test_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    check_errors(test_file, "pyright", res.stdout.decode("utf-8"), res.returncode)
