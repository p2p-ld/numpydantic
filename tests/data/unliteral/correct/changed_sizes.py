"""
After an operation that changes sizes, unliteral keeps the number of dimensions,
and unshape gives the array literal sizes again
"""

from typing import Literal as L
from typing import assert_type

import numpy as np

from numpydantic import NDArray, Shape, unliteral, unshape

f = np.float64


def takes_2d(array: NDArray[Shape[int, int], f]) -> None: ...


def pads(a: NDArray[Shape[L[3], L[4]], f]) -> None:
    padded: NDArray[Shape[int, int], f] = unliteral(np.pad(a, 1))
    takes_2d(padded)

    sized: NDArray[Shape[L[5], L[6]], f] = unshape(padded)
    assert_type(sized, NDArray[Shape[L[5], L[6]], f])
