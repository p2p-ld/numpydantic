import subprocess
from pathlib import Path

import pyright
import pytest

DATA_DIR = Path(__file__).parent / "data"

pytestmark = pytest.mark.typechecking


@pytest.mark.parametrize(
    "test_file",
    [
        pytest.param(p, id=p.stem)
        for p in sorted((DATA_DIR / "stub" / "correct").glob("*.py"))
    ],
)
def test_pyright(test_file: Path) -> None:
    """The mypy examples should pass static type checking with pyright too."""
    res = pyright.run(
        str(test_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    err = res.stderr.decode("utf-8")
    out = res.stdout.decode("utf-8")
    assert res.returncode == 0, f"stderr:\n{err}\n\nstdout:\n{out}"
