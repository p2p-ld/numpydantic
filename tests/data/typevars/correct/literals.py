"""Shape typevars bound or constrained to literal sizes, given sizes that fit"""

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
    x2: NDArray[Shape[L[2]], F],
    x3: NDArray[Shape[L[3]], F],
    x4: NDArray[Shape[L[4]], F],
    x5: NDArray[Shape[L[5]], F],
    x7: NDArray[Shape[L[7]], F],
) -> None:
    # sizes equal to a literal bound pass
    three(x3, x3)
    # equal sizes within a multi-value literal bound pass
    two_or_three(x2, x2)
    # equal sizes within a union of literals bound pass
    union(x3, x3)
    # equal sizes that fit the first literal constraint pass
    constrained(x3, x3)
    # equal sizes that fit the second literal constraint pass
    constrained(x4, x4)
    # equal sizes within the first, multi-value literal constraint pass
    literals(x2, x2)
    # equal sizes that fit the second literal constraint pass
    literals(x5, x5)
    # any size fits an int constraint, as long as the sizes are equal
    int_or_3(x7, x7)
