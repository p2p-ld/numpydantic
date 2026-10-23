"""
Mypy stub-only and plugin-based tests!
"""

from __future__ import annotations

import operator
import re
from pathlib import Path

import mypy.api
import pytest
from annotated_types import Interval
from mypy.options import Options
from mypy.subtypes import is_subtype
from mypy.test.typefixture import TypeFixture
from mypy.types import LiteralType

# from numpydantic import ndarray, update_ndarray_stub
from numpydantic.interface import Interface
from numpydantic.mypy.plugin_ import CONFIGFILE_KEY, MypyPluginOptions
from numpydantic.mypy.ranges import RangeValue
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
        assert returncode != 0, "Should not have passed mypy!\n" + msg


@pytest.mark.parametrize(
    "interfaces_string", ["numpy,dask,zarr", " numpy, dask ,zarr,"]
)
def test_parse_ini(interfaces_string, tmp_path):
    """
    The mypy plugin is configurable from INI files as well as pyproject.toml
    """
    cfg_str = f"[{CONFIGFILE_KEY}]\ninterfaces={interfaces_string}\n"
    cfg = tmp_path / "mypy.ini"
    cfg.write_text(cfg_str)

    options = Options()
    options.config_file = str(cfg)

    plugin_options = MypyPluginOptions.from_options(options)
    assert plugin_options.interfaces == ["numpy", "dask", "zarr"]


# ---------------------------------------------------------------------------
# Range directionality
# ---------------------------------------------------------------------------


def _range(low: int | None, high: int | None) -> RangeValue:
    return RangeValue(Interval(ge=low, le=high))


@pytest.mark.parametrize(
    "subtype,supertype,expected",
    [
        pytest.param(_range(2, 3), _range(1, 5), True, id="narrow-in-wide"),
        pytest.param(_range(1, 5), _range(2, 3), False, id="wide-in-narrow"),
        pytest.param(_range(2, 3), _range(2, 3), True, id="same"),
        pytest.param(_range(1, 3), _range(2, 5), False, id="overlapping"),
        pytest.param(_range(2, None), _range(1, None), True, id="open-high-in-wide"),
        pytest.param(_range(1, None), _range(2, None), False, id="open-high-in-narrow"),
        pytest.param(_range(None, 3), _range(None, 5), True, id="open-low-in-wide"),
        pytest.param(_range(None, 5), _range(None, 3), False, id="open-low-in-narrow"),
        pytest.param(_range(2, None), _range(2, 3), False, id="open-in-closed"),
        pytest.param(_range(2, 3), _range(None, None), True, id="closed-in-unbounded"),
        pytest.param(3, _range(2, 4), True, id="exact-in-range"),
        pytest.param(5, _range(2, 4), False, id="exact-outside-range"),
        pytest.param(_range(2, 4), 3, True, id="range-in-exact-permissive"),
        pytest.param(_range(2, 4), 5, False, id="range-in-exact-outside"),
    ],
)
def test_range_subtype(
    subtype: int | RangeValue, supertype: int | RangeValue, expected: bool
) -> None:
    """
    A range is a subtype of another range only if it fits within it.

    Checked with mypy's own subtype check, since the direction depends on mypy
    comparing ``subtype.value == supertype.value`` in that order.
    Comparing a range and an exact size can't tell the direction,
    so a range is a subtype of an exact size within it (range-in-exact-permissive).
    """
    fallback = TypeFixture().a
    left = LiteralType(subtype, fallback=fallback)
    right = LiteralType(supertype, fallback=fallback)
    assert is_subtype(left, right) is expected


@pytest.mark.parametrize(
    "left,right,equal",
    [
        pytest.param(_range(2, 3), _range(1, 5), True, id="narrow-wide"),
        pytest.param(_range(1, 5), _range(2, 3), False, id="wide-narrow"),
        pytest.param(3, _range(1, 5), True, id="int-left"),
        pytest.param(_range(1, 5), 3, True, id="int-right"),
        pytest.param(9, _range(1, 5), False, id="int-left-outside"),
        pytest.param(_range(1, 5), 9, False, id="int-right-outside"),
        pytest.param(_range(0, 5), True, False, id="bool"),
    ],
)
def test_range_comparisons(
    left: int | RangeValue, right: int | RangeValue, equal: bool
) -> None:
    """
    RangeValue's own comparisons, without mypy.
    The order of the operands matters, so they are compared with ``operator``
    rather than ``==``, which linters like to reorder.
    """
    assert operator.eq(left, right) is equal
    # __ne__ agrees with __eq__ rather than comparing as strings
    assert operator.ne(left, right) is not equal


def test_range_exclusive_bounds() -> None:
    """Exclusive bounds become the equivalent inclusive bounds"""
    value = RangeValue(Interval(gt=1, lt=5))
    assert (value.low, value.high) == (2, 4)
    assert str(value) == "2-4"


def test_range_cache(tmp_path: Path) -> None:
    """
    Ranges still work when mypy reuses its cache for the module that defines them,
    where they would otherwise be loaded as plain strings
    """
    (tmp_path / "mypy.ini").write_text(
        "[mypy]\n"
        "plugins = numpydantic.mypy\n"
        f"mypy_path = {MYPY_DIR / 'correct'}\n"
    )
    # compare a wide range and a narrow range from the cached module
    user = tmp_path / "user.py"
    user.write_text("from shape_range_subset import narrow, x\n\ny = x\ny = narrow()\n")
    args = [
        str(user),
        "--config-file",
        str(tmp_path / "mypy.ini"),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
    ]

    for _ in range(2):
        stdout, stderr, returncode = mypy.api.run(args)
        assert returncode == 0, stdout + stderr
        # only change the user module, so the ranges module can come from the cache
        user.write_text(user.read_text() + "# edited\n")
