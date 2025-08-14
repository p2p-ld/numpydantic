from typing import Any
from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[L["1"]], Any] = np.empty((1,))


def a_func(array: np.typing.NDArray) -> None:
    pass


if isinstance(x, np.ndarray):
    a_func(x)
