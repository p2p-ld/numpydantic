from typing import Literal as L

import numpy as np

from numpydantic import AnyDims, NDArray, Shape


def f(x: NDArray[Shape[L[3], *AnyDims], np.float64]) -> None: ...


f(np.zeros((4, 5)))
