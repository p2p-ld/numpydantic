"""Exact sizes can be used where ranges that hold all of them are expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def sizes() -> NDArray[Shape[L[2, 4]], np.uint8]:
    return np.zeros((2,), dtype=np.uint8)


def seven() -> NDArray[Shape[L[7]], np.uint8]:
    return np.zeros((7,), dtype=np.uint8)


# 2 and 4 are both in the range
x: NDArray[Shape[Range[L[2], L[4]]], np.uint8] = sizes()
# 7 is in one of the ranges in a union
y: NDArray[Shape[Range[L[1], L[2]] | Range[L[6], L[7]]], np.uint8] = seven()
