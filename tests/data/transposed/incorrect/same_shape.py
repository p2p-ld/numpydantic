"""A transposed array no longer has its original shape"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, transposed


def f(a: NDArray[Shape[L[3], L[4]], np.float64]) -> None:
    b: NDArray[Shape[L[3], L[4]], np.float64] = transposed(a.T)  # E: Incompatible types in assignment
    print(b)
