"""A range, or an exact size, can be used where a wider range is expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape

Wide = NDArray[Shape[Range[L[1], L[5]], Range[L[1], int]], np.uint8]


def narrow() -> NDArray[Shape[Range[L[2], L[3]], Range[L[2], int]], np.uint8]:
    return np.zeros((2, 2), dtype=np.uint8)


def exact() -> NDArray[Shape[L[3], L[9]], np.uint8]:
    return np.zeros((3, 9), dtype=np.uint8)


x: Wide = narrow()
y: Wide = exact()
