"""The NDArray[] with no args is equivalent to np.ndarray"""

import numpy as np

from numpydantic import NDArray


def a_func() -> np.ndarray:
    return np.zeros((3, 3))


x: NDArray = a_func()
