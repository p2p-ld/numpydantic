"""
bare ndarray annotations can be assigned to NDArray when not created from a constructor
"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> np.ndarray:
    return np.array([[1, 2, 3], [1, 2, 3]])


x: NDArray[Shape[L[10], L[10]], float] = a_func()
