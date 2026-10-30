"""Shape typevars in each form of shape, given sizes that agree or aren't known"""

from typing import Annotated as A
from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Range, Shape

N = TypeVar("N", bound=int)
F = np.float64


def labelled(a: NDArray[Shape[A[N, "x"], A[N, "y"]], F]) -> None: ...
def repeated(a: NDArray[Shape[N, ...], F]) -> None: ...
def unpacked(a: NDArray[Shape[L[2], *tuple[N, ...]], F]) -> None: ...
def triple(a: NDArray[Shape[N, N, N], F]) -> None: ...


def use(
    m33: NDArray[Shape[L[3], L[3]], F],
    c333: NDArray[Shape[L[3], L[3], L[3]], F],
    t233: NDArray[Shape[L[2], L[3], L[3]], F],
    c3i3: NDArray[Shape[L[3], int, L[3]], F],
    c3r3: NDArray[Shape[L[3], Range[L[2], L[4]], L[3]], F],
) -> None:
    # a typevar in labelled dimensions with equal sizes passes
    labelled(m33)
    # a repeated typevar with the same size in every dimension passes
    repeated(c333)
    # a typevar repeated after a fixed size passes when the repeated sizes are equal
    unpacked(t233)
    # a size that isn't known, int, can't disagree with the equal sizes around it
    triple(c3i3)
    # a range of sizes can't disagree with the equal sizes around it
    triple(c3r3)
