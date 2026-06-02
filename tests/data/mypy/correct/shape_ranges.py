"""Inclusive shape ranges"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def ranged() -> NDArray[Shape[L["2-4, 2-*, *-4"]], np.float32]:
    return np.zeros((3, 5, 2), dtype=np.float32)


arr = ranged()
