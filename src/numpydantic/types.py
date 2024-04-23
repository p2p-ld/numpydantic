"""
Types for numpydantic

Note that these are types as in python typing types, not classes.
"""

# ruff: noqa: D102

from typing import Any, Protocol, Tuple, Union, runtime_checkable

from nptyping import DType

ShapeType = Union[Tuple[int, ...], Any]
DtypeType = Union[str, type, Any, DType]


@runtime_checkable
class NDArrayType(Protocol):
    """A protocol for describing types that should be considered ndarrays"""

    @property
    def dtype(self) -> DtypeType: ...

    @property
    def shape(self) -> ShapeType: ...

    def __getitem__(self, key: Union[int, slice]) -> "NDArrayType": ...

    def __setitem__(self, key: Union[int, slice], value: "NDArrayType"): ...
