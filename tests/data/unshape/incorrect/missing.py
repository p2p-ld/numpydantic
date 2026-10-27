"""Sizes only known to be int can't be assigned to a literal shape without unshape"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def size() -> int:
    return 3


x: NDArray[Shape[L[3], L[3]], np.float64] = np.zeros((size(), size()))
