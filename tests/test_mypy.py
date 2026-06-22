"""
Mypy stub-only and plugin-based tests!
"""

from __future__ import annotations

import re
from pathlib import Path

import mypy.api
import pytest

# from numpydantic import ndarray, update_ndarray_stub
from numpydantic.interface import Interface
from numpydantic.testing import ValidationCase
from numpydantic.testing.cases import MYPY_CASES

DATA_DIR = Path(__file__).parent / "data"
MYPY_DIR = DATA_DIR / "mypy"

pytestmark = pytest.mark.typechecking

_ERROR_MARKER = re.compile(r"#\s*E:\s*(.+?)\s*$")
_MYPY_ERROR_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):.*?error:\s*(?P<msg>.*)$")

TYPED_INTERFACES: list[str] = [
    i.name for i in Interface.interfaces(with_disabled=True) if i.typing
]


@pytest.fixture(scope="module")
def all_interfaces_config(tmp_path_factory) -> Path:
    """Nothing to see here, just a blank file without the plugins configured"""
    config_path = tmp_path_factory.mktemp("mypy") / "pyproject.toml"
    ifaces = "\n".join(['"' + iname + '",' for iname in TYPED_INTERFACES])

    cfg = f"""
    [tool.mypy]
    plugins = ["numpydantic.mypy"]
    
    [tool.numpydantic.mypy]
    interfaces = [
      {ifaces}
    ]
    """

    config_path.write_text(cfg)
    return config_path


@pytest.mark.parametrize(
    "test_file",
    [
        pytest.param(p, id=p.stem)
        for p in sorted((DATA_DIR / "general" / "correct").glob("*.py"))
    ],
)
def test_mypy_handwritten_correct(test_file: Path, mypy_cache_dir):
    """The mypy examples should pass static type checking with only the stub"""
    res = mypy.api.run(
        [
            str(test_file),
            "--cache-dir",
            str(mypy_cache_dir),
        ]
    )
    assert res[2] == 0, res[0]
    assert "Success: no issues found in 1 source file" in res[0]
    assert res[1] == ""


@pytest.mark.parametrize(
    "test_file",
    [
        pytest.param(p, id=p.stem)
        for p in sorted((DATA_DIR / "general" / "incorrect").glob("*.py"))
    ],
)
def test_mypy_handwritten_incorrect(test_file: Path, mypy_cache_dir):
    """The mypy examples should pass static type checking with only the stub"""
    res = mypy.api.run(
        [
            str(test_file),
            "--cache-dir",
            str(mypy_cache_dir),
        ]
    )
    assert res[2] != 0, res[0] + res[1]
    assert res[1] == ""


@pytest.mark.parametrize(
    "test_file",
    [pytest.param(p, id=p.stem) for p in sorted((MYPY_DIR / "correct").glob("*.py"))],
)
def test_mypy_correct(test_file: Path, mypy_cache_dir) -> None:
    """Files in correct/ must pass mypy with zero errors."""
    stdout, stderr, returncode = mypy.api.run(
        [str(test_file), "--cache-dir", str(mypy_cache_dir)]
    )
    assert (
        "Success: no issues found in 1 source file" in stdout
    ), f"stdout:\n{stdout}\nstderr:\n{stderr}"
    assert stderr == ""
    assert returncode == 0


@pytest.mark.parametrize("test_file", sorted((MYPY_DIR / "incorrect").glob("*.py")))
def test_mypy_incorrect(test_file: Path, mypy_cache_dir) -> None:
    """Files in incorrect/ must fail mypy, and ``# E:`` markers must match."""
    stdout, stderr, returncode = mypy.api.run(
        [str(test_file), "--cache-dir", str(mypy_cache_dir)]
    )
    assert (
        returncode != 0
    ), f"expected mypy to fail but it succeeded:\n{stdout}\nstderr:\n{stderr}"


# ---------------------------------------------------------------------------
# Combinatorial generator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mypy_cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mypy_cache")


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c.id, marks=c.pytest_marks) for c in MYPY_CASES]
)
def test_mypy_generated(
    case: ValidationCase, tmp_path, request, all_interfaces_config, mypy_cache_dir
) -> None:
    test_file = tmp_path / (request.node.name + ".py")
    mypy_str = case.emit_mypy_source()
    test_file.write_text(mypy_str)

    stdout, stderr, returncode = mypy.api.run(
        [
            str(test_file),
            "--config-file",
            str(all_interfaces_config),
            "--ignore-missing-imports",
            "--cache-dir",
            str(mypy_cache_dir),
            "--show-traceback",
            "--follow-untyped-imports",
        ]
    )
    stdout = str(stdout).replace(str(test_file), test_file.stem)

    # make a version of the test file with line numbers
    numbered = "\n".join(
        [
            f"{str(i).rjust(2)}: " + line
            for i, line in enumerate(mypy_str.splitlines(), 1)
        ]
    )

    msg = (
        f"path:{test_file}\nsource:\n{numbered}\n\n"
        + ("-" * 50)
        + f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )

    if case.passes:
        assert returncode == 0, "Should have passed mypy!\n" + msg
    else:
        assert returncode != 0, "Should not have pased mypy!\n" + msg
