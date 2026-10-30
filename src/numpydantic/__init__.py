# ruff: noqa: F401
# ruff: noqa: I001
# ruff: noqa: D104

from typing import Literal as L

from numpydantic.ndarray import NDArray
from numpydantic.shape import Range, Repeat, Shape, transposed, unliteral, unshape
from numpydantic.annotation import NDArraySchema
from numpydantic import dtype

S = Shape
R = Range
Rep = Repeat
ND = NDArray
figurative = unliteral  # come on lol how could i not

__all__ = [
    "NDArray",
    "NDArraySchema",
    "Range",
    "Repeat",
    "Shape",
    "S",
    "L",
    "ND",
    "R",
    "Rep",
    "dtype",
    "figurative",
    "transposed",
    "unliteral",
    "unshape",
]
