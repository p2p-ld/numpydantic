"""A range can't be used where a narrower range is expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def wide() -> NDArray[Shape[Range[L[1], L[5]]], np.uint8]:
    return np.zeros((3,), dtype=np.uint8)


x: NDArray[Shape[Range[L[2], L[3]]], np.uint8] = wide()  # E: Incompatible types
