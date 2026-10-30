"""Shape typevars given different sizes, in each kind of callable"""

from typing import Generic, TypeVar, assert_type
from typing import Literal as L

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape, unshape

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
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
    return diagonal(matrix)


def twice(matrix: NDArray[Shape[K, K], F]) -> NDArray[Shape[K], F]:
    return passed_on(matrix)


def wrong_shape(matrix: NDArray[Shape[K, K], F]) -> NDArray[Shape[K, K], F]:
    # the inner result is shaped (K,), so it isn't the declared (K, K)
    return diagonal(matrix)  # E: Incompatible return value type


def other_typevar(
    matrix: NDArray[Shape[K, K], F], other: NDArray[Shape[J], F]
) -> NDArray[Shape[J], F]:
    # K and J stay separate typevars: the declared result makes the inner call's N J,
    # so the (K, K) argument doesn't fit
    return diagonal(matrix)  # E: Argument 1 to "diagonal" has incompatible type "ndarray[tuple[K, K], dtype[float64]]"; expected "ndarray[tuple[J, J], dtype[float64]]"


class Holder:
    def square(self, x: NDArray[Shape[K, K], F]) -> None: ...


class Pair(Generic[N]):
    def __init__(self, a: NDArray[Shape[N], F], b: NDArray[Shape[N], F]) -> None: ...


class Model(BaseModel, Generic[N]):
    vector: NDArray[Shape[N], F]
    matrix: NDArray[Shape[N, L[2]], F]


def use(
    x3: NDArray[Shape[L[3]], F],
    x4: NDArray[Shape[L[4]], F],
    m33: NDArray[Shape[L[3], L[3]], F],
    m34: NDArray[Shape[L[3], L[4]], F],
    m42: NDArray[Shape[L[4], L[2]], F],
) -> NDArray[Shape[L[4]], F]:
    # two arguments that give a typevar different sizes fail
    pair(x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"
    # one argument that gives a typevar a different size in each dimension fails
    diagonal(m34)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "matrix"
    # with two typevars, the one given different sizes fails: N is 4 in vector but 3 in matrix
    first_row(x4, m34)  # E[mypy]: Shape typevar "N" is 4 in "vector" but 3 in "matrix"
    # the call is fine, but the size that comes back out through two levels is 3, not 4
    assert_type(twice(m33), NDArray[Shape[L[4]], F])  # E: Expression is of type
    # a method's own typevar given different sizes fails
    Holder().square(m34)  # E[mypy]: Shape typevar "K" is both 3 and 4 in "x"
    # a generic class's constructor arguments with different sizes fail
    Pair(x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"
    # model fields that give a typevar different sizes fail
    Model(vector=x3, matrix=m42)  # E[mypy]: Shape typevar "N" is 3 in "vector" but 4 in "matrix"
    # fields that don't fit a parametrized model's size fail
    Model[L[3]](vector=x4, matrix=m42)  # E: Argument "vector" to "Model" has incompatible type
    # a result declared with a different size from the arguments fails:
    # the declared result makes N 4, so the 3x3 argument doesn't fit
    return diagonal(m33)  # E: Argument 1 to "diagonal" has incompatible type
