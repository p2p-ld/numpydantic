from typing import Literal as L
from typing import assert_type, reveal_type

import numpy as np

from numpydantic import NDArray, Shape

Square = NDArray[Shape[L[3], L[3]], np.float64]

loaded = np.load("square.npy")
reveal_type(loaded)

x: Square = loaded
reveal_type(x)
assert_type(x, Square)

table = np.loadtxt("square.txt")
y: Square = table
assert_type(y, Square)
