from typing import Any

import numpy as np

from numpydantic import NDArray, dtype

# incorrect literal dtypes
w: NDArray[Any, np.uint16] = np.ones((1, 2, 3), dtype=np.uint8)

# builtin aliases for their numpy equivalents
x: NDArray[Any, int] = np.ones((1, 2, 3), dtype=np.float64)

# Unions too
y: NDArray[Any, np.uint8 | np.uint16] = np.ones((1, 2, 3), dtype=np.float64)

# And numpydantic's alias types
z: NDArray[Any, dtype.Integer] = np.ones((1, 2, 3), dtype=np.float64)
