from typing import reveal_type

import numpy as np

from numpydantic import NDArray, Shape, L

x = np.zeros((2, 3))
reveal_type(x)
y: NDArray[Shape[L[2], L[3]]] = x
