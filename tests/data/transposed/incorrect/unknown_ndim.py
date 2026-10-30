"""Reversing an unknown number of int sizes still has an unknown number of sizes"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Repeat, Shape, transposed


def f(a: NDArray[Shape[L[3], *Repeat], np.float64]) -> None:
    b: NDArray[Shape[L[3], L[4]], np.float64] = transposed(a.T)  # E: Incompatible types in assignment
    print(b)
