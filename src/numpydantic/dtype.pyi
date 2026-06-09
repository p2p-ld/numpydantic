"""Static type stubs for :mod:`numpydantic.dtype`.

The compound dtype names (``Integer``, ``Float``, ``Complex``, ``Number``,
``Int``, ``SignedInteger``, ``UnsignedInteger``, ``Floating``) are tuples of
numpy generic types at runtime. We re-declare them here as union ``TypeAlias``
values so they can be used as the dtype argument of
:class:`numpydantic.NDArray`.
"""

from datetime import datetime
from typing import TypeAlias

import numpy as np

ShapeExpression: TypeAlias = str
StructureExpression: TypeAlias = str
DType: TypeAlias = np.generic | StructureExpression | tuple["DType", ...]
ShapeTuple: TypeAlias = tuple[int, ...]

Bool: TypeAlias = np.bool_
Obj: TypeAlias = np.object_
Object: TypeAlias = np.object_
Datetime64: TypeAlias = np.datetime64
Inexact: TypeAlias = np.inexact

Int8: TypeAlias = np.int8
Int16: TypeAlias = np.int16
Int32: TypeAlias = np.int32
Int64: TypeAlias = np.int64
Byte: TypeAlias = np.byte
Short: TypeAlias = np.short
IntC: TypeAlias = np.intc
IntP: TypeAlias = np.intp
Int_: TypeAlias = np.int_
UInt8: TypeAlias = np.uint8
UInt16: TypeAlias = np.uint16
UInt32: TypeAlias = np.uint32
UInt64: TypeAlias = np.uint64
UByte: TypeAlias = np.ubyte
UShort: TypeAlias = np.ushort
UIntC: TypeAlias = np.uintc
UIntP: TypeAlias = np.uintp
UInt: TypeAlias = np.uint
ULongLong: TypeAlias = np.ulonglong
LongLong: TypeAlias = np.longlong
Timedelta64: TypeAlias = np.timedelta64

SignedInteger: TypeAlias = np.int8 | np.int16 | np.int32 | np.int64 | np.short
UnsignedInteger: TypeAlias = np.uint8 | np.uint16 | np.uint32 | np.uint64 | np.ushort
Integer: TypeAlias = SignedInteger | UnsignedInteger
Int: TypeAlias = Integer

Float16: TypeAlias = np.float16
Float32: TypeAlias = np.float32
Float64: TypeAlias = np.float64
Half: TypeAlias = np.half
Single: TypeAlias = np.single
Double: TypeAlias = np.double
LongDouble: TypeAlias = np.longdouble
Float: TypeAlias = np.float16 | np.float32 | np.float64 | np.single | np.double
Floating: TypeAlias = Float

Complex64: TypeAlias = np.complex64
Complex128: TypeAlias = np.complex128
CSingle: TypeAlias = np.csingle
CDouble: TypeAlias = np.cdouble
CLongDouble: TypeAlias = np.clongdouble
Complex: TypeAlias = (
    np.complex64 | np.complex128 | np.csingle | np.cdouble | np.clongdouble
)

Flexible: TypeAlias = np.flexible
Void: TypeAlias = np.void
Character: TypeAlias = np.character
Bytes: TypeAlias = np.bytes_
Str: TypeAlias = np.str_
String: TypeAlias = np.str_
Unicode: TypeAlias = np.str_

Datetime: TypeAlias = datetime | np.datetime64

Number: TypeAlias = Integer | Float | Complex
