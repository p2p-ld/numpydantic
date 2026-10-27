"""A trailing ... repeats the dimension before it"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def repeated() -> NDArray[Shape[L[1], L[2], ...], np.float64]:
    return np.zeros((1, 2, 2, 2))


def one() -> NDArray[Shape[L[1], L[2], ...], np.float64]:
    return np.zeros((1, 2))


def ranged() -> NDArray[Shape[Range[L[2], L[4]], ...], np.float64]:
    return np.zeros((2, 3, 4))


x = repeated()
y = one()
z = ranged()
