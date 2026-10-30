"""A transposed array with an Any dtype no longer has its original shape"""

from typing import Any
from typing import Literal as L

from numpydantic import NDArray, Shape, transposed


def f(a: NDArray[Shape[L[3], L[4]], Any]) -> None:
    b: NDArray[Shape[L[3], L[4]], Any] = transposed(a.T)  # E: Incompatible types in assignment
    print(b)
