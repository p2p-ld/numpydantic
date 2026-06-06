"""NDArrays can be assigned to a bare numpy annotation always"""

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> NDArray[Shape["3, 3"], float]:
    return np.zeros((3, 3))


x: np.ndarray = a_func()
