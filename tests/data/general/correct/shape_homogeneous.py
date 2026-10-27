# /// typecheck
# [pyright]
# requires = { numpy = ">=2.5,<2.6" }
# ///

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def homogeneous() -> NDArray[Shape[L[2], ...], np.float64]:
    return np.zeros((2, 2, 2))


def none() -> NDArray[Shape[L[2], ...], np.float64]:
    return np.zeros(())


x = homogeneous()
y = none()
