from typing import Literal as L
from typing import assert_type

import numpy as np

from numpydantic import NDArray, Repeat, Shape, unshape


def at_least_one() -> NDArray[Shape[L[3], *Repeat], np.float64]:
    return unshape(np.zeros((3, 4, 5)))


def first_dim_only() -> NDArray[Shape[L[3], *Repeat], np.float64]:
    return unshape(np.zeros((3,)))


def anything() -> NDArray[Shape[*Repeat], np.float64]:
    return np.zeros(())


x = at_least_one()
assert_type(x, NDArray[Shape[L[3], *Repeat], np.float64])
y = first_dim_only()
assert_type(y, NDArray[Shape[L[3], *Repeat], np.float64])
z = anything()
