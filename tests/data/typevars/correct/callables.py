"""Shape typevars given sizes that agree, in each kind of callable"""

from typing import Generic, TypeVar, assert_type
from typing import Literal as L

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape, unshape

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
F = np.float64


def pair(a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]) -> NDArray[Shape[N], F]:
    return a


def diagonal(matrix: NDArray[Shape[N, N], F]) -> NDArray[Shape[N], F]:
    return unshape(np.diagonal(matrix))


def first_row(
    vector: NDArray[Shape[N], F], matrix: NDArray[Shape[N, K], F]
) -> NDArray[Shape[K], F]:
    return matrix[0]


def passed_on(matrix: NDArray[Shape[K, K], F]) -> NDArray[Shape[K], F]:
    # a typevar can be passed on to another function's typevar,
    # and the inner result keeps it, so it's the declared result
    return diagonal(matrix)


def twice(matrix: NDArray[Shape[K, K], F]) -> NDArray[Shape[K], F]:
    # passing it on through two levels keeps it too
    return passed_on(matrix)


class Holder:
    def square(self, x: NDArray[Shape[K, K], F]) -> None: ...


class Pair(Generic[N]):
    def __init__(self, a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]) -> None: ...


class Model(BaseModel, Generic[N]):
    vector: NDArray[Shape[N], F]
    matrix: NDArray[Shape[N, L[2]], F]


def use(
    x3: NDArray[Shape[L[3]], F],
    y3: NDArray[Shape[L[3]], F],
    m33: NDArray[Shape[L[3], L[3]], F],
    m34: NDArray[Shape[L[3], L[4]], F],
    m32: NDArray[Shape[L[3], L[2]], F],
) -> NDArray[Shape[L[3]], F]:
    # two arguments that give a typevar the same size pass
    pair(x3, y3)
    # one argument that gives a typevar the same size in each dimension passes
    diagonal(m33)
    # two typevars each keep their own size: N is 3 in both arguments, and K is 4
    first_row(x3, m34)
    # the size given at the call comes back out through two levels of typevars
    assert_type(twice(m33), NDArray[Shape[L[3]], F])
    # a method's own typevar is matched within its arguments
    Holder().square(m33)
    # a generic class's constructor arguments that agree pass
    Pair(x3, y3)
    # model fields that give a typevar the same size pass
    Model(vector=x3, matrix=m32)
    # fields that fit a parametrized model's size pass
    Model[L[3]](vector=x3, matrix=m32)
    # the result has the size the arguments gave the typevar
    return diagonal(m33)
