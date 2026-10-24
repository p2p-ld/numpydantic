from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def make_range() -> NDArray[Shape[Range[L[3], L[5]], Range[L[3], L[5]]]]:
    return np.ones((4, 4))


x: NDArray[Shape[Range[L[6], L[8]], Range[L[6], L[8]]]] = make_range()
