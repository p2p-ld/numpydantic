"""dtype argument must be a numpy generic, Any, or a tuple/union thereof."""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def f() -> NDArray[Shape[L["3"]], list]:  # E: invalid dtype
    return np.zeros((3,), dtype=np.int32)
