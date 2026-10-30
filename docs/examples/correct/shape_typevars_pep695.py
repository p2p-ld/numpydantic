from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, unshape


def diagonal[N: int](
    matrix: NDArray[Shape[N, N], np.float64],
) -> NDArray[Shape[N], np.float64]:
    return unshape(np.diagonal(matrix))


def add[N: int](
    a: NDArray[Shape[N], np.float64], b: NDArray[Shape[N], np.float64]
) -> NDArray[Shape[N], np.float64]:
    return unshape(a + b)


def use(
    matrix: NDArray[Shape[L[3], L[3]], np.float64],
    x3: NDArray[Shape[L[3]], np.float64],
    y3: NDArray[Shape[L[3]], np.float64],
) -> None:
    d: NDArray[Shape[L[3]], np.float64] = diagonal(matrix)
    s: NDArray[Shape[L[3]], np.float64] = add(x3, y3)
