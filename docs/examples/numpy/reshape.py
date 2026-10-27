from typing import Literal as L, cast, reveal_type

import numpy as np

from numpydantic import NDArray, Shape

Wide = NDArray[Shape[L[3], L[4]], np.float64]
Long = NDArray[Shape[L[4], L[3]], np.float64]

x = cast(Wide, np.ones((3, 4)))
reveal_type(x)

y: Long = x.reshape(4, 3)
