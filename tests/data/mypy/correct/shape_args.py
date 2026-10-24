from typing import Any

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[1, 2, 3], Any] = np.zeros((1, 2, 3))
