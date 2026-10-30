"""
Parametrize type checker tests from a directory of example files.

A file can declare per-checker requirements in a block at its top,
written like inline script metadata (PEP 723) with TOML contents::

    # /// typecheck
    # [pyright]
    # requires = { numpy = ">=2.5" }
    # ///

The file is skipped for that checker when an installed version doesn't match.

A file that only one checker can handle is skipped for the other with a reason::

    # /// typecheck
    # [pyright]
    # skip = "only the mypy plugin refuses this"
    # ///

A known gap is marked as expected to fail, so fixing it fails the test
until the mark is removed::

    # /// typecheck
    # [mypy]
    # xfail = "mypy joins the list items before the plugin sees them"
    # ///

A file in an ``incorrect`` directory marks each line that should be refused::

    pair(a, b)  # E: Incompatible types
    pair(a, b)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"

``# E[mypy]:`` is refused only by mypy. Each marked line has to have an error,
and no other line can. mypy's error has to contain the marker's message.
pyright words its errors differently, so for pyright only the line is checked.
"""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from packaging.specifiers import SpecifierSet

Checker = Literal["mypy", "pyright"]

DATA_DIR = Path(__file__).parents[1] / "data"
SHARED_SUITES = ("general", "unshape", "unliteral", "transposed", "typevars")

_BLOCK = re.compile(
    r"^# /// typecheck$\s(?P<content>(^#(| .*)$\s)+)^# ///$", re.MULTILINE
)
_OPTIONS = {"requires", "skip", "xfail"}
_MARKER = re.compile(r"#\s*E(?:\[(?P<checker>\w+)\])?:\s*(?P<message>.*?)\s*$")
_ERROR = {
    "mypy": re.compile(r"^(?P<path>.+?):(?P<line>\d+): error: (?P<message>.*)$"),
    "pyright": re.compile(
        r"^\s*(?P<path>.+?):(?P<line>\d+):\d+ - error: (?P<message>.*)$"
    ),
}


def expected_errors(path: Path, checker: Checker) -> dict[int, str]:
    """The message of each ``# E:`` marker for a checker, by line number"""
    expected = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _MARKER.search(line)
        if match and match["checker"] in (None, checker):
            expected[number] = match["message"]
    return expected


def check_errors(path: Path, checker: Checker, output: str, returncode: int) -> None:
    """The errors a checker reported for a file are on the lines its markers expect"""
    expected = expected_errors(path, checker)
    # 1 means type errors were found, anything above it means the checker failed
    assert returncode == (1 if expected else 0), (
        f"{checker} exited with {returncode}, "
        f"but the markers expect errors on {sorted(expected)}:\n{output}"
    )

    errors: dict[int, list[str]] = {}
    for line in output.splitlines():
        match = _ERROR[checker].match(line)
        if match and Path(match["path"]).resolve() == path.resolve():
            errors.setdefault(int(match["line"]), []).append(match["message"])

    assert sorted(errors) == sorted(expected), (
        f"{checker} refused lines {sorted(errors)}, "
        f"but the markers expect {sorted(expected)}:\n{output}"
    )
    if checker == "mypy":
        for number, message in expected.items():
            assert any(message in error for error in errors[number]), (
                f"line {number} should be refused with {message!r}, "
                f"got {errors[number]}"
            )


def read_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the ``typecheck`` metadata block of a test file, if it has one"""
    match = _BLOCK.search(path.read_text())
    if match is None:
        return {}
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match["content"].splitlines(keepends=True)
    )
    metadata = tomllib.loads(content)
    for checker, options in metadata.items():
        if checker not in get_args(Checker) or not set(options) <= _OPTIONS:
            raise ValueError(
                f"{path}: unknown typecheck metadata [{checker}] {options!r}"
            )
    return metadata


def _marks(path: Path, checker: Checker) -> list[pytest.MarkDecorator]:
    options = read_metadata(path).get(checker, {})
    if "skip" in options:
        return [pytest.mark.skip(reason=f"{checker}: {options['skip']}")]
    requires = options.get("requires", {})
    marks = []
    if "xfail" in options:
        marks.append(
            pytest.mark.xfail(reason=f"{checker}: {options['xfail']}", strict=True)
        )
    for package, spec in requires.items():
        installed = version(package)
        if not SpecifierSet(spec).contains(installed, prereleases=True):
            marks.append(
                pytest.mark.skip(
                    reason=f"{checker} needs {package}{spec}, found {installed}"
                )
            )
    return marks


def typecheck_params(checker: Checker, *directories: Path) -> list[Any]:
    """One ``pytest.param`` per python file, with ids like ``suite/file``"""
    return [
        pytest.param(
            path,
            id=f"{directory.parent.name}/{path.stem}",
            marks=_marks(path, checker),
        )
        for directory in directories
        for path in sorted(directory.glob("*.py"))
    ]


def shared_params(
    outcome: Literal["correct", "incorrect"], checker: Checker
) -> list[Any]:
    """Cases in every suite that all type checkers should agree on"""
    return typecheck_params(
        checker, *(DATA_DIR / suite / outcome for suite in SHARED_SUITES)
    )
