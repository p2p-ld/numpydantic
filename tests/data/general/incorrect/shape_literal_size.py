"""
A wrong size is refused: with numpy>=2.5 because the literal sizes differ,
and before that because numpy only knows the sizes are int
"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape[L[3], L[3]], np.float64] = np.zeros((4, 4))  # E: Incompatible types in assignment
