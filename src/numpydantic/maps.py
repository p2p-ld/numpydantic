"""
Maps from one value system to another
"""

from datetime import datetime
from typing import Any

import numpy as np

from numpydantic import dtype as dt

np_to_python = {
    Any: Any,
    np.number: float,
    np.object_: Any,
    np.bool_: bool,
    np.integer: int,
    np.byte: bytes,
    np.bytes_: bytes,
    np.datetime64: datetime,
    **{n: int for n in dt.Integer},
    **{n: float for n in dt.Float},
    **{n: complex for n in dt.Complex},
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

python_to_nptyping = {
    float: dt.Float,
    str: dt.String,
    int: dt.Int,
    bool: dt.Bool,
    complex: dt.Complex,
}
"""Map from python types to nptyping types"""
