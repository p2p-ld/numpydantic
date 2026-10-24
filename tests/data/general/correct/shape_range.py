from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Range, Shape


def ranged() -> (
    NDArray[Shape[Range[L[2], L[4]], Range[L[2], int], Range[int, L[4]]], np.float32]
):
    return np.zeros((3, 5, 2), dtype=np.float32)


arr = ranged()
