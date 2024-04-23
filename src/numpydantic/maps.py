"""
Maps from one value system to another
"""

from datetime import datetime
from typing import Any

import numpy as np
from nptyping import Bool, Float, Int, String

np_to_python = {
    Any: Any,
    np.number: float,
    np.object_: Any,
    np.bool_: bool,
    np.integer: int,
    np.byte: bytes,
    np.bytes_: bytes,
    np.datetime64: datetime,
    **{
        n: int
        for n in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.short,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.uint,
        )
    },
    **{
        n: float
        for n in (
            np.float16,
            np.float32,
            np.floating,
            np.float32,
            np.float64,
            np.single,
            np.double,
            np.float_,
        )
    },
    **{n: str for n in (np.character, np.str_, np.string_, np.unicode_)},
}
"""Map from python types to numpy"""


flat_to_nptyping = {
    "float": "Float",
    "float32": "Float32",
    "double": "Double",
    "float64": "Float64",
    "long": "LongLong",
    "int64": "Int64",
    "int": "Int",
    "int32": "Int32",
    "int16": "Int16",
    "short": "Short",
    "int8": "Int8",
    "uint": "UInt",
    "uint32": "UInt32",
    "uint16": "UInt16",
    "uint8": "UInt8",
    "uint64": "UInt64",
    "numeric": "Number",
    "text": "String",
    "utf": "Unicode",
    "utf8": "Unicode",
    "utf_8": "Unicode",
    "string": "Unicode",
    "str": "Unicode",
    "ascii": "String",
    "bool": "Bool",
    "isodatetime": "Datetime64",
    "AnyType": "Any",
    "object": "Object",
}
"""Map from NWB-style flat dtypes to nptyping types"""

python_to_nptyping = {float: Float, str: String, int: Int, bool: Bool}
"""Map from python types to nptyping types"""
