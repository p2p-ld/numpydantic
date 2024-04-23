"""
Types for numpydantic

Note that these are types as in python typing types, not classes.
"""

# ruff: noqa: D102

from typing import Any, Protocol, Tuple, runtime_checkable

import numpy as np
from nptyping import DType

ShapeType = Tuple[int, ...] | Any
DtypeType = np.dtype | str | type | Any | DType


@runtime_checkable
class NDArrayType(Protocol):
    """A protocol for describing types that should be considered ndarrays"""

    @property
    def dtype(self) -> DtypeType: ...

    @property
    def shape(self) -> ShapeType: ...

    def __getitem__(self, key: int | slice) -> "NDArrayType": ...

    def __setitem__(self, key: int | slice, value: "NDArrayType"): ...
