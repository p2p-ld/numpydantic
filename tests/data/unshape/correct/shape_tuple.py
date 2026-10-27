from typing import Any, assert_type
from typing import Literal as L

import numpy as np

from numpydantic import NDArray, unshape

x: NDArray[tuple[L[1], L[2], L[3]], Any] = unshape(np.zeros((1, 2, 3)))
assert_type(x, NDArray[tuple[L[1], L[2], L[3]], Any])
