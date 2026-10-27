from typing import Literal as L
from typing import cast, reveal_type, TypeAlias

import numpy as np

from numpydantic import NDArray, Shape

Square: TypeAlias = NDArray[Shape[L[3], L[3]], np.float64]

square = np.zeros((3, 3))
reveal_type(square)

x = cast(Square, square)
reveal_type(x)
