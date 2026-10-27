from typing import Literal as L
from typing import assert_type, reveal_type, TypeAlias

import numpy as np

from numpydantic import NDArray, Shape, unliteral, unshape

Array: TypeAlias = NDArray[Shape[L[3], L[4]], np.float64]


def pads(a: Array) -> NDArray[Shape[int, int], np.float64]:
    # numpy says np.pad keeps the sizes it was given, which is wrong
    wrong = np.pad(a, 1)
    reveal_type(wrong)

    # unliteral keeps the number of dimensions and forgets the sizes
    padded = unliteral(wrong)
    reveal_type(padded)
    assert_type(padded, NDArray[Shape[int, int], np.float64])
    return padded
