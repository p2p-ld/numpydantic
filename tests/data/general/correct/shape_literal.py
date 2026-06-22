from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[L[1], L[2], L[3]]] = np.zeros((1, 2, 3))
