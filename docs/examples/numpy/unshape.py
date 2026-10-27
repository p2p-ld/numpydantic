from typing import Literal as L, reveal_type
from typing import assert_type, TypeAlias

import numpy as np

from numpydantic import NDArray, Shape, unshape

Square: TypeAlias = Shape[L[3], L[3]]
FloatSquare: TypeAlias = NDArray[Square, np.float64]
IntSquare: TypeAlias = NDArray[Square, np.uint8]


def size() -> int:
    return 3


zeros = np.zeros((3, 3))
reveal_type(zeros)

x = unshape(zeros)
reveal_type(x)

# no good, dtype mismatch
y: IntSquare = unshape(zeros)

# fine!
z: FloatSquare = unshape(zeros)
reveal_type(z)
