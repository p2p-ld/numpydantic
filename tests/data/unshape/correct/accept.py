"""
unshape lets an array whose sizes are only known to be ``int``
be used where a literal shape is expected,
and takes on that shape
"""

from typing import Literal as L
from typing import assert_type

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape, unshape

Square = NDArray[Shape[L[3], L[3]], np.float64]


def size() -> int:
    return 3


def takes_square(array: Square) -> None:
    assert_type(array, Square)


def make_square() -> Square:
    return unshape(np.zeros((size(), size())))


class Model(BaseModel):
    array: Square


x: Square = unshape(np.zeros((size(), size())))
assert_type(x, Square)
takes_square(unshape(np.zeros((size(), size()))))
model = Model(array=unshape(np.zeros((size(), size()))))
assert_type(model.array, Square)
assert_type(make_square(), Square)

# unshape doesn't check the shape, so a wrong annotation is accepted too
wrong: NDArray[Shape[L[5]], np.float64] = unshape(np.zeros((size(), size())))
assert_type(wrong, NDArray[Shape[L[5]], np.float64])
