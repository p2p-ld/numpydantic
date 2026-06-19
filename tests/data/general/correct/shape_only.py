from typing import Literal

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[Literal[3], Literal[3]]] = np.ndarray((3, 3))
