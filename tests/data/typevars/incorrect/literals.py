"""
Shape typevars bound or constrained to literal sizes,
given a size that doesn't fit, or different sizes that each fit
"""

from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Shape

Three = TypeVar("Three", bound=L[3])
TwoOrThree = TypeVar("TwoOrThree", bound=L[2, 3])
Union23 = TypeVar("Union23", bound=L[2] | L[3])
C = TypeVar("C", L[3], L[4])
Literals = TypeVar("Literals", L[1, 2], L[5])
IntOr3 = TypeVar("IntOr3", int, L[3])
F = np.float64


def three(a: NDArray[Shape[Three], F], b: NDArray[Shape[Three], F]) -> None: ...
def two_or_three(
    a: NDArray[Shape[TwoOrThree], F], b: NDArray[Shape[TwoOrThree], F]
) -> None: ...
def union(a: NDArray[Shape[Union23], F], b: NDArray[Shape[Union23], F]) -> None: ...
def constrained(a: NDArray[Shape[C], F], b: NDArray[Shape[C], F]) -> None: ...
def literals(
    a: NDArray[Shape[Literals], F], b: NDArray[Shape[Literals], F]
) -> None: ...
def int_or_3(a: NDArray[Shape[IntOr3], F], b: NDArray[Shape[IntOr3], F]) -> None: ...


def use(
    x1: NDArray[Shape[L[1]], F],
    x2: NDArray[Shape[L[2]], F],
    x3: NDArray[Shape[L[3]], F],
    x4: NDArray[Shape[L[4]], F],
    x5: NDArray[Shape[L[5]], F],
    x7: NDArray[Shape[L[7]], F],
) -> None:
    # sizes outside a literal bound fail
    three(x4, x4)  # E: Value of type variable "Three" of "three" cannot be "Literal[4]"
    # sizes outside a multi-value literal bound fail
    two_or_three(x4, x4)  # E: Value of type variable "TwoOrThree" of "two_or_three" cannot be "Literal[4]"
    # different sizes fail, even when both are within a multi-value literal bound
    two_or_three(x2, x3)  # E[mypy]: Shape typevar "TwoOrThree" is 2 in "a" but 3 in "b"
    # sizes outside a union of literals bound fail
    union(x4, x4)  # E: Value of type variable "Union23" of "union" cannot be "Literal[4]"
    # different sizes fail, even when both are within a union of literals bound
    union(x2, x3)  # E[mypy]: Shape typevar "Union23" is 2 in "a" but 3 in "b"
    # sizes outside the literal constraints fail
    constrained(x5, x5)  # E: Value of type variable "C" of "constrained" cannot be "Literal[5]"
    # sizes that fit different literal constraints fail
    constrained(x3, x4)  # E: Value of type variable "C" of "constrained" cannot be
    # sizes outside the multi-value literal constraints fail
    literals(x3, x3)  # E: Value of type variable "Literals" of "literals" cannot be "Literal[3]"
    # different sizes fail, even when both are within one multi-value literal constraint
    literals(x1, x2)  # E[mypy]: Shape typevar "Literals" is 1 in "a" but 2 in "b"
    # different sizes fail, even when both fit an int constraint
    int_or_3(x3, x7)  # E[mypy]: Shape typevar "IntOr3" is 3 in "a" but 7 in "b"
