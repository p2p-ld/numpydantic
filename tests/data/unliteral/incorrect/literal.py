"""unliteral's result can't be used where literal sizes are expected"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape, unliteral


def f(a: NDArray[Shape[L[3], L[4]], np.float64]) -> None:
    b: NDArray[Shape[L[3], L[4]], np.float64] = unliteral(a)
    print(b)
