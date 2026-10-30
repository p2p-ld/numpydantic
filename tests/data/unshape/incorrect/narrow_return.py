"""A shape given by a return type after unshape is refused by a different shape"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, unshape


def size() -> int:
    return 3


def takes_4x4(array: NDArray[Shape[L[4], L[4]], np.float64]) -> None: ...


def make_square() -> NDArray[Shape[L[3], L[3]], np.float64]:
    return unshape(np.zeros((size(), size())))


takes_4x4(make_square())  # E: incompatible type
