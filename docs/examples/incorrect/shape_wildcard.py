from typing import Literal as L
from typing import reveal_type

import numpy as np

from numpydantic import NDArray, Repeat, Shape

AT_LEAST_2D = NDArray[Shape[int, int, *Repeat]]


def returns_atleast2d() -> AT_LEAST_2D:
    return np.ones((2, 3, 4))


# two or more dimensions!
x: AT_LEAST_2D = np.ones((1,))

# `int` dimensions can't be narrowed without cast,
# even if the narrowing is within the range
# (since the rhs could be a shape that doesn't fit the narrowing)
y: NDArray[Shape[L[2], L[3], L[4]]] = returns_atleast2d()

reveal_type(x)
reveal_type(y)
