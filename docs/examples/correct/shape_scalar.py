from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def make_array() -> NDArray[Shape[L[1], L[2], L[3]], np.uint8]:
    return np.ones((1, 2, 3), dtype=np.uint8)


x: NDArray[Shape[L[1], L[2], L[3]], np.uint8] = make_array()
