# ruff: noqa: F401
# ruff: noqa: I001
# ruff: noqa: D104

from typing import Literal as L

from numpydantic.ndarray import NDArray
from numpydantic.shape import AnyDims, Range, Shape
from numpydantic.annotation import NDArraySchema
from numpydantic import dtype

S = Shape
R = Range
ND = NDArray

__all__ = [
    "AnyDims",
    "NDArray",
    "NDArraySchema",
    "Range",
    "Shape",
    "S",
    "L",
    "ND",
    "R",
    "dtype",
]
