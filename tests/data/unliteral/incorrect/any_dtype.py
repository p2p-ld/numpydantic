"""unliteral's result with an Any dtype can't be used as literal sizes"""

from typing import Any
from typing import Literal as L

from numpydantic import NDArray, Shape, unliteral


def f(a: NDArray[Shape[L[3], L[4]], Any]) -> None:
    b: NDArray[Shape[L[3], L[4]], Any] = unliteral(a)  # E: Incompatible types in assignment
    print(b)
