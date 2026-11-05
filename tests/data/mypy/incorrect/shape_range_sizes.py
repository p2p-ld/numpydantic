"""Exact sizes can't be used where ranges without all of them are expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def sizes() -> NDArray[Shape[L[2, 5]], np.uint8]:
    return np.zeros((2,), dtype=np.uint8)


def four() -> NDArray[Shape[L[4]], np.uint8]:
    return np.zeros((4,), dtype=np.uint8)


# 5 isn't in the range
x: NDArray[Shape[Range[L[2], L[4]]], np.uint8] = sizes()  # E: Incompatible types in assignment
# 4 is between the ranges in a union, but in neither of them
y: NDArray[Shape[Range[L[1], L[2]] | Range[L[6], L[7]]], np.uint8] = four()  # E: Incompatible types in assignment
