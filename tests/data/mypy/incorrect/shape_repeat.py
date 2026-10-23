"""A trailing ... repeats the dimension before it, rather than allowing anything"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def repeated() -> NDArray[Shape[L[1], L[2], ...], np.float64]:
    return np.zeros((1, 2, 3))  # E: Incompatible return value type
