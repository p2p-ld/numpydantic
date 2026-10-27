from typing import Literal as L
from typing import assert_type

import numpy as np

from numpydantic import NDArray, Shape, unshape

x: NDArray[Shape[L[1], L[2], L[3]]] = unshape(np.zeros((1, 2, 3)))
assert_type(x, NDArray[Shape[L[1], L[2], L[3]]])
