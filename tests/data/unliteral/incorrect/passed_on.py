"""An array passed on after unliteral can't be used where literal sizes are expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, unliteral


def takes_3x4(array: NDArray[Shape[L[3], L[4]], np.float64]) -> None: ...


def f(a: NDArray[Shape[L[3], L[4]], np.float64]) -> None:
    b = unliteral(a)
    takes_3x4(b)
