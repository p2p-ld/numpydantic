"""Can be used with just a shape with an implicit Any dtype"""

from typing import Any
from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape


def a_func() -> NDArray[Shape[L[3], L[3]], Any]:
    return np.zeros((3, 3))


x: NDArray[Shape[L[3], L[3]]] = a_func()
