"""
MIT License

Copyright (c) 2023 Ramon Hagenaars

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from numpydantic.vendor.nptyping.assert_isinstance import assert_isinstance
from numpydantic.vendor.nptyping.error import (
    InvalidArgumentsError,
    InvalidDTypeError,
    InvalidShapeError,
    InvalidStructureError,
    NPTypingError,
)
from numpydantic.vendor.nptyping.ndarray import NDArray
from numpydantic.vendor.nptyping.package_info import __version__

# don't import unnecessarily since we don't use it
# from numpydantic.vendor.nptyping.pandas_.dataframe import DataFrame
from numpydantic.vendor.nptyping.recarray import RecArray
from numpydantic.vendor.nptyping.shape import Shape
from numpydantic.vendor.nptyping.shape_expression import (
    normalize_shape_expression,
    validate_shape_expression,
)
from numpydantic.vendor.nptyping.structure import Structure
from numpydantic.vendor.nptyping.typing_ import (
    Bool,
    Byte,
    Bytes,
    CDouble,
    CFloat,
    Character,
    CLongDouble,
    CLongFloat,
    Complex,
    Complex64,
    Complex128,
    ComplexFloating,
    CSingle,
    Datetime64,
    Double,
    DType,
    Flexible,
    Float,
    Float16,
    Float32,
    Float64,
    Floating,
    Half,
    Inexact,
    Int,
    Int8,
    Int16,
    Int32,
    Int64,
    IntC,
    Integer,
    IntP,
    LongComplex,
    LongDouble,
    LongFloat,
    LongLong,
    Number,
    Object,
    Short,
    SignedInteger,
    Single,
    SingleComplex,
    String,
    Timedelta64,
    UByte,
    UInt,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UIntC,
    UIntP,
    ULongLong,
    Unicode,
    UnsignedInteger,
    UShort,
    Void,
)

__all__ = [
    "NDArray",
    "RecArray",
    "assert_isinstance",
    "validate_shape_expression",
    "normalize_shape_expression",
    "NPTypingError",
    "InvalidArgumentsError",
    "InvalidShapeError",
    "InvalidStructureError",
    "InvalidDTypeError",
    "Shape",
    "Structure",
    "__version__",
    "DType",
    "Number",
    "Bool",
    "Bool8",
    "Object",
    "Object0",
    "Datetime64",
    "Integer",
    "SignedInteger",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Byte",
    "Short",
    "IntC",
    "IntP",
    "Int0",
    "Int",
    "LongLong",
    "Timedelta64",
    "UnsignedInteger",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UByte",
    "UShort",
    "UIntC",
    "UIntP",
    "UInt0",
    "UInt",
    "ULongLong",
    "Inexact",
    "Floating",
    "Float16",
    "Float32",
    "Float64",
    "Half",
    "Single",
    "Double",
    "Float",
    "LongDouble",
    "LongFloat",
    "ComplexFloating",
    "Complex64",
    "Complex128",
    "CSingle",
    "SingleComplex",
    "CDouble",
    "Complex",
    "CFloat",
    "CLongDouble",
    "CLongFloat",
    "LongComplex",
    "Flexible",
    "Void",
    "Void0",
    "Character",
    "Bytes",
    "String",
    "Bytes0",
    "Unicode",
    "Str0",
    "DataFrame",
]
