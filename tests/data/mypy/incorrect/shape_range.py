"""Shape outside of a Range"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def make_array() -> NDArray[Shape[L[1], L[2], L[3]], np.uint8]:
    return np.zeros((1, 2, 3), dtype=np.uint8)


x: NDArray[
    Shape[Range[L[1], L[3]], Range[L[1], L[4]], Range[L[10], L[20]]], np.uint8
] = make_array()  # E: Incompatible types in assignment
