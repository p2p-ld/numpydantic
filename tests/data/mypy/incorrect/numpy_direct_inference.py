"""
Shape can be inferred from constructors directly
"""

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape["10, 10"], float] = np.zeros((3, 3))
