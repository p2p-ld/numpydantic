"""
Types for numpydantic

Note that these are types as in python typing types, not classes.
"""

# ruff: noqa: D102

from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

from numpydantic.vendor.nptyping import DType

ShapeType: TypeAlias = tuple[int, ...] | Any
DtypeType: TypeAlias = str | type | Any | DType

_T_Shape = TypeVar("_T_Shape", bound=ShapeType)
_T_Dtype = TypeVar("_T_Dtype", bound=DtypeType)


@runtime_checkable
class NDArrayType(Protocol[_T_Shape, _T_Dtype]):
    """A protocol for describing types that should be considered ndarrays"""

    @property
    def dtype(self) -> DtypeType: ...

    @property
    def shape(self) -> ShapeType: ...

    def __getitem__(self, key: int | slice) -> "NDArrayType": ...

    def __setitem__(self, key: int | slice, value: "NDArrayType"): ...
