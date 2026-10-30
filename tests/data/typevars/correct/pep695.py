# /// typecheck
# [mypy]
# requires = { python = ">=3.12" }
# [pyright]
# requires = { python = ">=3.12" }
# ///
"""Shape typevars declared with python 3.12's syntax, given sizes that agree"""

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
    y3: NDArray[Shape[L[3]], F],
    x2: NDArray[Shape[L[2]], F],
    m33: NDArray[Shape[L[3], L[3]], F],
    m44: NDArray[Shape[L[4], L[4]], F],
    m32: NDArray[Shape[L[3], L[2]], F],
    f3: NDArray[Shape[L[5], L[5], L[3]], F],
) -> None:
    # two arguments that give a function's typevar the same size pass
    pair(x3, y3)
    # one argument that gives a typevar the same size in each dimension passes
    diagonal(m33)
    # the size given at the call comes back out through two levels of typevars
    assert_type(passed_on(m33), NDArray[Shape[L[3]], F])
    # equal sizes that fit a literal constraint pass
    channels(f3, f3)
    # equal sizes within a multi-value literal constraint pass
    one_to_five(x2, x2)
    # sizes within a range bound pass
    small(m44)
    # a generic class's constructor arguments that agree pass
    Pair(x3, y3)
    # model fields that give a typevar the same size pass
    Model(vector=x3, matrix=m32)
