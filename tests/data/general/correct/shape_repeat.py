# /// typecheck
# [pyright]
# requires = { numpy = ">=2.5,<2.6" }
# ///

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Repeat, Shape


def at_least_one() -> NDArray[Shape[L[3], *Repeat], np.float64]:
    return np.zeros((3, 4, 5))


def first_dim_only() -> NDArray[Shape[L[3], *Repeat], np.float64]:
    return np.zeros((3,))


def anything() -> NDArray[Shape[*Repeat], np.float64]:
    return np.zeros(())


def twos(x: NDArray[Shape[L[3], *Repeat[L[2]]], np.float64]) -> None: ...


x = at_least_one()
y = first_dim_only()
z = anything()
# every repeated dimension is the given size
twos(np.zeros((3, 2, 2)))
# including none of them
twos(np.zeros((3,)))
