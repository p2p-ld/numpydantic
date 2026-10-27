"""
The deprecated string forms of shapes still type check with the mypy plugin
"""

from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape["1, 2, 3"]] = np.zeros((1, 2, 3))


def ranged() -> NDArray[Shape[L["2-4, 2-*, *-4"]], np.float32]:
    return np.zeros((3, 5, 2), dtype=np.float32)


def labelled() -> NDArray[Shape[L["3 x, 4 y, * z"]], np.uint8]:
    return np.zeros((3, 4, 7), dtype=np.uint8)


def anything_after() -> NDArray[Shape["3, ..."], np.uint8]:
    return np.zeros((3, 4, 7), dtype=np.uint8)


a = ranged()
b = labelled()
c = anything_after()
