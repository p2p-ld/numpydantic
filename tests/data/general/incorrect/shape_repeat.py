from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Repeat, Shape


def f(x: NDArray[Shape[L[3], *Repeat], np.float64]) -> None: ...
def twos(x: NDArray[Shape[L[3], *Repeat[L[2]]], np.float64]) -> None: ...


f(np.zeros((4, 5)))  # E: incompatible type
# a repeated dimension that isn't the given size fails
twos(np.zeros((3, 2, 4)))  # E: incompatible type
