"""Shape typevars bound or constrained to ranges, given sizes that fit"""

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
    x4: NDArray[Shape[L[4]], F],
    x6: NDArray[Shape[L[6]], F],
) -> None:
    # equal sizes at the low end of a range bound pass, since ranges are inclusive
    ranged(x2, x2)
    # equal sizes at the high end of a range bound pass
    ranged(x4, x4)
    # equal sizes within the first range constraint pass
    ranges(x1, x1)
    # equal sizes within the second range constraint pass
    ranges(x6, x6)
