"""
Types for numpydantic

Note that these are types as in python typing types, not classes.
"""

from typing import Any, Protocol, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
from nptyping import DType


ShapeType = Tuple[int, ...] | Any
DtypeType = np.dtype | str | type | Any | DType


@runtime_checkable
class NDArrayType(Protocol):

    @property
    def dtype(self) -> DtypeType: ...

    @property
    def shape(self) -> ShapeType: ...

    def __getitem__(self, key: int | slice) -> "NDArrayType": ...

    def __setitem__(self, key: int | slice, value: "NDArrayType"): ...
