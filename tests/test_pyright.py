import subprocess
from pathlib import Path

import numpy as np
import pyright
import pytest

DATA_DIR = Path(__file__).parent / "data"

pytestmark = [pytest.mark.typechecking, pytest.mark.pyright]

# These assign numpy constructors like np.zeros((3, 3)) to literal shapes,
# which needs numpy>=2.5 to infer the literal shape.
# TODO: rewrite these as boundary-crossing cases rather than skipping them
NEEDS_SHAPE_INFERENCE = {
    "numpy_getitem",
    "numpy_model_field",
    "shape_anydims",
    "shape_homogeneous",
    "shape_labels",
    "shape_literal",
    "shape_tuple",
}
OLD_NUMPY = np.lib.NumpyVersion(np.__version__) < "2.5.0"


def _correct_param(path: Path) -> pytest.param:
    marks = []
    if path.stem in NEEDS_SHAPE_INFERENCE:
        marks.append(
            pytest.mark.skipif(OLD_NUMPY, reason="needs numpy>=2.5 shape inference")
        )
    return pytest.param(path, id=path.stem, marks=marks)


@pytest.mark.parametrize(
    "test_file",
    [
        _correct_param(p)
        for p in sorted((DATA_DIR / "general" / "correct").glob("*.py"))
    ],
)
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


@pytest.mark.parametrize(
    "test_file",
    [
        pytest.param(p, id=p.stem)
        for p in sorted((DATA_DIR / "general" / "incorrect").glob("*.py"))
    ],
)
def test_pyright_incorrect(test_file: Path) -> None:
    """Pyright fails when we expect it to"""
    res = pyright.run(
        str(test_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    err = res.stderr.decode("utf-8")
    out = res.stdout.decode("utf-8")
    assert res.returncode != 0, f"stderr:\n{err}\n\nstdout:\n{out}"
