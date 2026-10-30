"""
Shape can be inferred from constructors directly
"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[L[10], L[10]], float] = np.zeros((3, 3))  # E: Incompatible types in assignment
