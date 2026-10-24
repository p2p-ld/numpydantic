"""NDArrays can be assigned to a bare numpy annotation always"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> NDArray[Shape[L[3], L[3]], float]:
    return np.zeros((3, 3))


x: np.ndarray = a_func()
