# /// typecheck
# [mypy]
# requires = { python = ">=3.12" }
# [pyright]
# requires = { python = ">=3.12" }
# ///
"""Shape typevars declared with python 3.12's syntax, given sizes that disagree"""

from typing import Literal as L
from typing import assert_type

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Range, Shape, unshape

F = np.float64


def pair[N: int](a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]) -> NDArray[Shape[N], F]:
    return a


def diagonal[N: int](matrix: NDArray[Shape[N, N], F]) -> NDArray[Shape[N], F]:
    return unshape(np.diagonal(matrix))


def passed_on[K: int](matrix: NDArray[Shape[K, K], F]) -> NDArray[Shape[K], F]:
    return diagonal(matrix)


def channels[RGB: (L[3], L[4])](
    a: NDArray[Shape[int, int, RGB], F], b: NDArray[Shape[int, int, RGB], F]
) -> None: ...


def one_to_five[N: (L[1], L[2, 3, 4, 5])](
    a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]
) -> None: ...


def small[S: Range[L[1], L[4]]](a: NDArray[Shape[S, S], F]) -> None: ...


class Pair[N: int]:
    def __init__(self, a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]) -> None: ...


class Model[N: int](BaseModel):
    vector: NDArray[Shape[N], F]
    matrix: NDArray[Shape[N, L[2]], F]


def use(
    x3: NDArray[Shape[L[3]], F],
    x4: NDArray[Shape[L[4]], F],
    x2: NDArray[Shape[L[2]], F],
    m33: NDArray[Shape[L[3], L[3]], F],
    m34: NDArray[Shape[L[3], L[4]], F],
    m55: NDArray[Shape[L[5], L[5]], F],
    m42: NDArray[Shape[L[4], L[2]], F],
    f3: NDArray[Shape[L[5], L[5], L[3]], F],
    f4: NDArray[Shape[L[5], L[5], L[4]], F],
) -> None:
    # two arguments that give a function's typevar different sizes fail
    pair(x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"
    # one argument that gives a typevar a different size in each dimension fails
    diagonal(m34)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "matrix"
    # the call is fine, but the size that comes back out through two levels is 3, not 4
    assert_type(passed_on(m33), NDArray[Shape[L[4]], F])  # E: Expression is of type
    # sizes that fit different literal constraints fail
    channels(f3, f4)  # E: Value of type variable "RGB" of "channels" cannot be
    # different sizes fail, even when both are within one multi-value literal constraint
    one_to_five(x2, x3)  # E[mypy]: Shape typevar "N" is 2 in "a" but 3 in "b"
    # sizes outside a range bound fail
    small(m55)  # E[mypy]: Value of type variable "S" of "small" cannot be "Literal[5]"
    # a generic class's constructor arguments with different sizes fail
    Pair(x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"
    # model fields that give a typevar different sizes fail
    Model(vector=x3, matrix=m42)  # E[mypy]: Shape typevar "N" is 3 in "vector" but 4 in "matrix"
