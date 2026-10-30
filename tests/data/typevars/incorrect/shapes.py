"""Shape typevars in each form of shape, given different sizes"""

from typing import Annotated as A
from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Range, Repeat, Shape

N = TypeVar("N", bound=int)
F = np.float64


def labelled(a: NDArray[Shape[A[N, "x"], A[N, "y"]], F]) -> None: ...
def repeated(a: NDArray[Shape[N, ...], F]) -> None: ...
def unpacked(a: NDArray[Shape[L[2], *tuple[N, ...]], F]) -> None: ...
def repeat(a: NDArray[Shape[L[2], *Repeat[N]], F]) -> None: ...
def at_least_one(a: NDArray[Shape[L[2], N, *Repeat[N]], F]) -> None: ...
def triple(a: NDArray[Shape[N, N, N], F]) -> None: ...


def use(
    m34: NDArray[Shape[L[3], L[4]], F],
    c344: NDArray[Shape[L[3], L[4], L[4]], F],
    t234: NDArray[Shape[L[2], L[3], L[4]], F],
    c3i4: NDArray[Shape[L[3], int, L[4]], F],
    c3r4: NDArray[Shape[L[3], Range[L[2], L[4]], L[4]], F],
) -> None:
    # a typevar in labelled dimensions with different sizes fails
    labelled(m34)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # a repeated typevar with different sizes fails
    repeated(c344)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # a typevar repeated after a fixed size fails when the repeated sizes differ
    unpacked(t234)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # the same, written with Repeat
    repeat(t234)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # a typevar both fixed and repeated, the spelling of Shape[L[2], N, ...]
    # that type checkers accept, fails when its sizes differ
    at_least_one(t234)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # a size that isn't known, int, doesn't hide the known sizes disagreeing
    triple(c3i4)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
    # a range of sizes doesn't hide the known sizes disagreeing
    triple(c3r4)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "a"
