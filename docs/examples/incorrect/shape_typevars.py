from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Shape, unshape

N = TypeVar("N", bound=int)


def diagonal(matrix: NDArray[Shape[N, N], np.float64]) -> NDArray[Shape[N], np.float64]:
    return unshape(np.diagonal(matrix))


def add(
    a: NDArray[Shape[N], np.float64], b: NDArray[Shape[N], np.float64]
) -> NDArray[Shape[N], np.float64]:
    return unshape(a + b)


def use(
    matrix: NDArray[Shape[L[3], L[4]], np.float64],
    x3: NDArray[Shape[L[3]], np.float64],
    x4: NDArray[Shape[L[4]], np.float64],
) -> None:
    # one argument that gives N different sizes
    diagonal(matrix)
    # two arguments that give N different sizes
    add(x3, x4)
