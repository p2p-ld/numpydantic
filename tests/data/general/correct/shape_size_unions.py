# /// typecheck
# [pyright]
# requires = { numpy = ">=2.5,<2.6" }
# ///
"""A dimension with several exact sizes, written as one Literal or a union"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

F = np.float64


def several(a: NDArray[Shape[L[3, 5], L[2]], F]) -> None: ...
def union(a: NDArray[Shape[L[3] | L[5], L[2]], F]) -> None: ...
def wider(a: NDArray[Shape[L[3, 4, 5], L[2]], F]) -> None: ...


def use(
    exact: NDArray[Shape[L[5], L[2]], F],
    narrower: NDArray[Shape[L[3, 5], L[2]], F],
) -> None:
    # either size passes
    several(np.zeros((3, 2)))
    several(np.zeros((5, 2)))
    # either size passes when the sizes are written as a union
    union(np.zeros((3, 2)))
    union(np.zeros((5, 2)))
    # an array annotated with one of the sizes passes
    several(exact)
    # an array with some of the sizes passes where more are allowed
    wider(narrower)
