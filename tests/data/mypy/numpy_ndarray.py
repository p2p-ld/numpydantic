"""
Testing that the bare type checks with a numpy array
"""

from typing import Any
from typing import Literal as L

import numpy as np

from numpydantic import NDArray

x: NDArray[L["1"], Any] = np.zeros((1,))


def a_func(array: np.typing.NDArray) -> None:
    pass


if isinstance(x, np.ndarray):
    a_func(x)
