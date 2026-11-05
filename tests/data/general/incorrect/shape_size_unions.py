"""A dimension with several exact sizes, written as one Literal or a union"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

F = np.float64


def several(a: NDArray[Shape[L[3, 5], L[2]], F]) -> None: ...
def union(a: NDArray[Shape[L[3] | L[5], L[2]], F]) -> None: ...
def narrower(a: NDArray[Shape[L[3, 4], L[2]], F]) -> None: ...


def use(
    exact: NDArray[Shape[L[4], L[2]], F],
    wider: NDArray[Shape[L[3, 4, 5], L[2]], F],
) -> None:
    # a size between the given sizes fails
    several(np.zeros((4, 2)))  # E: incompatible type
    # a size between the given sizes fails when they are written as a union
    union(np.zeros((4, 2)))  # E: incompatible type
    # an array annotated with a size that isn't one of them fails
    several(exact)  # E: incompatible type
    # an array that could have a size that isn't allowed fails
    narrower(wider)  # E: incompatible type
