"""
Shape typevars bound or constrained to ranges,
given a size outside them, or different sizes that each fit
"""

from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Range, Shape

R = TypeVar("R", bound=Range[L[2], L[4]])
Ranges = TypeVar("Ranges", Range[L[1], L[2]], Range[L[5], L[6]])
F = np.float64


def ranged(a: NDArray[Shape[R], F], b: NDArray[Shape[R], F]) -> None: ...
def ranges(a: NDArray[Shape[Ranges], F], b: NDArray[Shape[Ranges], F]) -> None: ...


def use(
    x1: NDArray[Shape[L[1]], F],
    x2: NDArray[Shape[L[2]], F],
    x3: NDArray[Shape[L[3]], F],
    x4: NDArray[Shape[L[4]], F],
    x5: NDArray[Shape[L[5]], F],
) -> None:
    # sizes just below a range bound fail
    ranged(x1, x1)  # E[mypy]: Value of type variable "R" of "ranged" cannot be "Literal[1]"
    # sizes just above a range bound fail
    ranged(x5, x5)  # E[mypy]: Value of type variable "R" of "ranged" cannot be "Literal[5]"
    # different sizes fail, even when both are within a range bound
    ranged(x3, x4)  # E[mypy]: Shape typevar "R" is 3 in "a" but 4 in "b"
    # sizes between the range constraints fail
    ranges(x3, x3)  # E[mypy]: Value of type variable "Ranges" of "ranges" cannot be "Literal[3]"
    # sizes that fit different range constraints fail
    ranges(x1, x5)  # E[mypy]: Value of type variable "Ranges" of "ranges" cannot be
    # different sizes fail, even when both are within one range constraint
    ranges(x1, x2)  # E[mypy]: Shape typevar "Ranges" is 1 in "a" but 2 in "b"
