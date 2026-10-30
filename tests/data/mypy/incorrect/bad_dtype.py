"""A dtype has to be a type, not a value"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def f() -> NDArray[Shape[L[3]], L[3]]:  # E: invalid dtype
    return np.zeros((3,), dtype=np.int32)
