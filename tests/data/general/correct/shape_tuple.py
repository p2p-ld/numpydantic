from typing import Any
from typing import Literal as L

import numpy as np

from numpydantic import NDArray

x: NDArray[tuple[L[1], L[2], L[3]], Any] = np.zeros((1, 2, 3))
