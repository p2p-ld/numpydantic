"""
generics are interpreted as generic even when they can be inferred
valid: ndarray[Any, Any] < ndarray[[3, 3], float64]
valid: ndarray[[10, 10], float64] < ndarray[Any, Any]

the second one has to be valid for compatibility reasons,
otherwise NDArray would be unusable in practice,
because this would be impossible

def some_array(data: list[list[int]]) -> NDArray[Shape[10, 10]]:
    return np.array(data)
"""

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> np.ndarray:
    return np.zeros((3, 3))


x: NDArray[Shape["10, 10"], float] = a_func()
