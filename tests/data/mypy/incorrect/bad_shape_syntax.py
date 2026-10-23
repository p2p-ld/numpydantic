"""Invalid shape expression: AnyDims can only be the last dimension."""

from typing import Literal as L

import numpy as np
from typing_extensions import Unpack

from numpydantic import AnyDims, NDArray, Shape


def f() -> (
    NDArray[Shape[Unpack[AnyDims], L[3]], np.uint8]
):  # E: must be the last item in a shape
    return np.zeros((1,), dtype=np.uint8)
