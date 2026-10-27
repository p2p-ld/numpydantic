# /// typecheck
# [pyright]
# requires = { numpy = ">=2.5,<2.6" }
# ///

from typing import Literal as L

import numpy as np

from numpydantic import AnyDims, NDArray, Shape


def at_least_one() -> NDArray[Shape[L[3], *AnyDims], np.float64]:
    return np.zeros((3, 4, 5))


def first_dim_only() -> NDArray[Shape[L[3], *AnyDims], np.float64]:
    return np.zeros((3,))


def anything() -> NDArray[Shape[*AnyDims], np.float64]:
    return np.zeros(())


x = at_least_one()
y = first_dim_only()
z = anything()
