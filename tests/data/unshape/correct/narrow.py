"""
After unshape, the annotation it's assigned to gives the array its shape,
which later checks then use
"""

from typing import Literal as L
from typing import assert_type

import numpy as np

from numpydantic import NDArray, Shape, unshape

Square = NDArray[Shape[L[3], L[3]], np.float64]


def size() -> int:
    return 3


def takes_square(array: Square) -> None: ...


def takes_2d(array: NDArray[Shape[int, int], np.float64]) -> None: ...


def make_square() -> Square:
    return unshape(np.zeros((size(), size())))


x: Square = unshape(np.zeros((size(), size())))
assert_type(x, Square)
takes_square(x)
takes_2d(x)

y = make_square()
assert_type(y, Square)
takes_square(y)
takes_2d(y)
