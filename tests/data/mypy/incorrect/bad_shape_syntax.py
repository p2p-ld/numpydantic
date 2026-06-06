"""Garbage shape expression."""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def f() -> (
    NDArray[Shape[L["this is not valid"]], np.uint8]
):  # E: not a valid shape expression
    return np.zeros((1,), dtype=np.uint8)
