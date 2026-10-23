"""Combining two ranges gives the wider one, which doesn't fit the narrower range"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def narrow() -> NDArray[Shape[Range[L[2], L[3]]], np.uint8]:
    return np.zeros((2,), dtype=np.uint8)


def wide() -> NDArray[Shape[Range[L[1], L[5]]], np.uint8]:
    return np.zeros((5,), dtype=np.uint8)


def either(flag: bool) -> NDArray[Shape[Range[L[2], L[3]]], np.uint8]:
    return narrow() if flag else wide()  # E: Incompatible return value type
