from typing import Literal as L
from typing import assert_type

import numpy as np

from numpydantic import NDArray, Shape, unshape


def homogeneous() -> NDArray[Shape[L[2], ...], np.float64]:
    return unshape(np.zeros((2, 2, 2)))


def none() -> NDArray[Shape[L[2], ...], np.float64]:
    return np.zeros(())


x = homogeneous()
assert_type(x, NDArray[Shape[L[2], ...], np.float64])
y = none()
