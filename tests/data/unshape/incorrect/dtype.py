"""unshape forgets the shape but keeps the dtype"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, unshape


def size() -> int:
    return 3


x: NDArray[Shape[L[3], L[3]], np.float64] = unshape(
    np.zeros((size(), size()), dtype=np.uint8)
)
