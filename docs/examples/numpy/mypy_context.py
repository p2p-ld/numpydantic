from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[L[3], L[3]], np.float64] = np.zeros((4, 4))
