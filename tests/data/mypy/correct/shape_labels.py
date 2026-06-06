"""Axis labels are documentation and don't affect type checking."""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def labelled() -> NDArray[Shape[L["3 x, 4 y, * z"]], np.uint8]:
    return np.zeros((3, 4, 7), dtype=np.uint8)


arr = labelled()
