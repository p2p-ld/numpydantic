"""
Parametrize type checker tests from a directory of example files.

A file can declare per-checker requirements in a block at its top,
written like inline script metadata (PEP 723) with TOML contents::

    # /// typecheck
    # [pyright]
    # requires = { numpy = ">=2.5" }
    # ///

The file is skipped for that checker when an installed version doesn't match.
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
SHARED_SUITES = ("general", "unshape", "unliteral", "transposed")

_BLOCK = re.compile(
    r"^# /// typecheck$\s(?P<content>(^#(| .*)$\s)+)^# ///$", re.MULTILINE
)
_OPTIONS = {"requires"}


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
    requires = read_metadata(path).get(checker, {}).get("requires", {})
    marks = []
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
