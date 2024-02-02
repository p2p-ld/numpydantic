from datetime import datetime
from typing import Any

import numpy as np

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
