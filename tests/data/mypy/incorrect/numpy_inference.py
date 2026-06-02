"""
bare ndarray annotations can't be assigned to NDArray
when they can be inferred from a constructor
"""

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> np.ndarray:
    return np.zeros((3, 3))


x: NDArray[Shape["10, 10"], float] = a_func()
