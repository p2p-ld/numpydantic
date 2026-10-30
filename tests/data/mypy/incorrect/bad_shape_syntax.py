"""Invalid shape expressions that only the mypy plugin refuses"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Repeat, Shape


# Repeat can only be the last dimension
def f() -> NDArray[Shape[*Repeat, L[3]], np.uint8]:  # E: must be the last item in a shape
    return np.zeros((1,), dtype=np.uint8)


# Repeat has to be unpacked, or it's a single dimension that is a tuple
def g() -> NDArray[Shape[L[3], Repeat], np.uint8]:  # E: Repeated dimensions have to be unpacked, like *Repeat
    return np.zeros((3,), dtype=np.uint8)
