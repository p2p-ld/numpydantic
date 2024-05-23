"""
Replacement of :mod:`nptyping.typing_`

In the transition away from using nptyping, we want to allow for greater
control of dtype specifications - like different precision modes, etc.
and allow for abstract specifications of dtype that can be checked across
interfaces.

This module also allows for convenient access to all abstract dtypes in a single
module, rather than needing to import each individually.

Some types like `Integer` are compound types - tuples of multiple dtypes.
Check these using ``in`` rather than ``==``. This interface will develop in future
versions to allow a single dtype check.
"""

import sys
from typing import Tuple, Union

if sys.version_info.minor >= 10:
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import numpy as np

ShapeExpression: TypeAlias = str
StructureExpression: TypeAlias = str
DType: TypeAlias = Union[np.generic, StructureExpression, Tuple["DType"]]
ShapeTuple: TypeAlias = Tuple[int, ...]

Bool = np.bool_
Obj = np.object_  # Obj is a common abbreviation and should be usable.
Object = np.object_
Datetime64 = np.datetime64
Inexact = np.inexact

Int8 = np.int8
Int16 = np.int16
Int32 = np.int32
Int64 = np.int64
Byte = np.byte
Short = np.short
IntC = np.intc
IntP = np.intp
Int_ = np.int_
UInt8 = np.uint8
UInt16 = np.uint16
UInt32 = np.uint32
UInt64 = np.uint64
UByte = np.ubyte
UShort = np.ushort
UIntC = np.uintc
UIntP = np.uintp
UInt = np.uint
ULongLong = np.ulonglong
LongLong = np.longlong
Timedelta64 = np.timedelta64
SignedInteger = (np.int8, np.int16, np.int32, np.int64, np.short)
UnsignedInteger = (np.uint8, np.uint16, np.uint32, np.uint64, np.ushort)
Integer = tuple([*SignedInteger, *UnsignedInteger])
"""All integer types"""
Int = Integer  # Int should translate to the "generic" int type.

Float16 = np.float16
Float32 = np.float32
Float64 = np.float64
Half = np.half
Single = np.single
Double = np.double
LongDouble = np.longdouble
LongFloat = np.longfloat
Float = (
    np.float_,
    np.float16,
    np.float32,
    np.float64,
    np.single,
    np.double,
)
Floating = Float

ComplexFloating = np.complexfloating
Complex64 = np.complex64
Complex128 = np.complex128
CSingle = np.csingle
SingleComplex = np.singlecomplex
CDouble = np.cdouble
CFloat = np.cfloat
CLongDouble = np.clongdouble
CLongFloat = np.clongfloat
Complex = (
    np.complex_,
    np.complexfloating,
    np.complex64,
    np.complex128,
    np.csingle,
    np.singlecomplex,
    np.cdouble,
    np.cfloat,
    np.clongdouble,
    np.clongfloat,
)

LongComplex = np.longcomplex
Flexible = np.flexible
Void = np.void
Character = np.character
Bytes = np.bytes_
Str = np.str_
String = np.string_
Unicode = np.unicode_

Number = tuple(
    [
        *Integer,
        *Float,
        *Complex,
    ]
)
