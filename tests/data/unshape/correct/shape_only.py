from typing import Literal, assert_type

import numpy as np

from numpydantic import NDArray, Shape, unshape

x: NDArray[Shape[Literal[3], Literal[3]]] = unshape(np.ndarray((3, 3)))
assert_type(x, NDArray[Shape[Literal[3], Literal[3]]])
