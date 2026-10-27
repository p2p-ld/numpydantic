from typing import Literal as L, reveal_type
from typing import assert_type, TypeAlias

import numpy as np

from numpydantic import NDArray, Shape, transposed, unshape

Wide: TypeAlias = NDArray[Shape[L[3], L[4]], np.float64]
Long: TypeAlias = NDArray[Shape[L[4], L[3]], np.float64]

x: Wide = unshape(np.zeros((3, 4)))

# oh no baby what is you doing!?
reveal_type(x.T)
reveal_type(np.transpose(x))

# transposed propagates the shape
reveal_type(transposed(x.T))
# mypy needs the assignment first
t = np.transpose(x)
reveal_type(transposed(t))

assert_type(transposed(x.T), Long)
assert_type(transposed(t), Long)

# Up to 5D arrays
z: NDArray[Shape[L[1], L[2], L[3], L[4]], np.float64] = unshape(np.zeros((1, 2, 3, 4)))
reveal_type(z)
reveal_type(transposed(z.T))
