from typing import Any, reveal_type

import numpy as np

from numpydantic import NDArray, dtype

# literal dtypes
w: NDArray[Any, np.uint8] = np.ones((1, 2, 3), dtype=np.uint8)

# builtin aliases for their numpy equivalents work
x: NDArray[Any, float] = np.ones((1, 2, 3), dtype=np.float64)

# Unions too
y: NDArray[Any, np.uint8 | np.uint16] = np.ones((1, 2, 3), dtype=np.uint8)

# And numpydantic's alias types
z: NDArray[Any, dtype.Integer] = np.ones((1, 2, 3), dtype=np.uint8)

reveal_type(w)
reveal_type(x)
reveal_type(y)
reveal_type(z)
