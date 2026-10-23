from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def make_array() -> NDArray[Shape[L[2], L[3], L[4]], np.uint8]:
    return np.ones((1, 2, 3), dtype=np.uint8)


x: NDArray[Shape[L[5], L[6], L[7]], np.uint8] = make_array()
