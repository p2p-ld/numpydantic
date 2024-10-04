import sys
from typing import TypeAlias, Union

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.dtype import Float, Integer, Number
from numpydantic.testing.helpers import ValidationCase

if sys.version_info.minor >= 10:
    from typing import TypeAlias

    YES_PIPE = True
else:
    from typing_extensions import TypeAlias

    YES_PIPE = False


class BasicModel(BaseModel):
    x: int


class BadModel(BaseModel):
    x: int


class SubClass(BasicModel):
    pass


RGB_UNION: TypeAlias = Union[
    NDArray[Shape["* x, * y"], Number],
    NDArray[Shape["* x, * y, 3 r_g_b"], Number],
    NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
]
NUMBER: TypeAlias = NDArray[Shape["*, *, *"], Number]
INTEGER: TypeAlias = NDArray[Shape["*, *, *"], Integer]
FLOAT: TypeAlias = NDArray[Shape["*, *, *"], Float]
STRING: TypeAlias = NDArray[Shape["*, *, *"], str]
MODEL: TypeAlias = NDArray[Shape["*, *, *"], BasicModel]
UNION_TYPE: TypeAlias = NDArray[Shape["*, *, *"], Union[np.uint32, np.float32]]
UNION_PIPE: TypeAlias = NDArray[Shape["*, *, *"], np.uint32 | np.float32]
DTYPE_CASES = [
    ValidationCase(dtype=float, passes=True),
    ValidationCase(dtype=int, passes=False),
    ValidationCase(dtype=np.uint8, passes=False),
    ValidationCase(annotation=NUMBER, dtype=int, passes=True),
    ValidationCase(annotation=NUMBER, dtype=float, passes=True),
    ValidationCase(annotation=NUMBER, dtype=np.uint8, passes=True),
    ValidationCase(annotation=NUMBER, dtype=np.float16, passes=True),
    ValidationCase(annotation=NUMBER, dtype=str, passes=False),
    ValidationCase(annotation=INTEGER, dtype=int, passes=True),
    ValidationCase(annotation=INTEGER, dtype=np.uint8, passes=True),
    ValidationCase(annotation=INTEGER, dtype=float, passes=False),
    ValidationCase(annotation=INTEGER, dtype=np.float32, passes=False),
    ValidationCase(annotation=INTEGER, dtype=str, passes=False),
    ValidationCase(annotation=FLOAT, dtype=float, passes=True),
    ValidationCase(annotation=FLOAT, dtype=np.float32, passes=True),
    ValidationCase(annotation=FLOAT, dtype=int, passes=False),
    ValidationCase(annotation=FLOAT, dtype=np.uint8, passes=False),
    ValidationCase(annotation=FLOAT, dtype=str, passes=False),
    ValidationCase(annotation=STRING, dtype=str, passes=True),
    ValidationCase(annotation=STRING, dtype=int, passes=False),
    ValidationCase(annotation=STRING, dtype=float, passes=False),
    ValidationCase(annotation=MODEL, dtype=BasicModel, passes=True),
    ValidationCase(annotation=MODEL, dtype=BadModel, passes=False),
    ValidationCase(annotation=MODEL, dtype=int, passes=False),
    ValidationCase(annotation=MODEL, dtype=SubClass, passes=True),
    ValidationCase(annotation=UNION_TYPE, dtype=np.uint32, passes=True),
    ValidationCase(annotation=UNION_TYPE, dtype=np.float32, passes=True),
    ValidationCase(annotation=UNION_TYPE, dtype=np.uint64, passes=False),
    ValidationCase(annotation=UNION_TYPE, dtype=np.float64, passes=False),
    ValidationCase(annotation=UNION_TYPE, dtype=str, passes=False),
]


DTYPE_IDS = [
    "float",
    "int",
    "uint8",
    "number-int",
    "number-float",
    "number-uint8",
    "number-float16",
    "number-str",
    "integer-int",
    "integer-uint8",
    "integer-float",
    "integer-float32",
    "integer-str",
    "float-float",
    "float-float32",
    "float-int",
    "float-uint8",
    "float-str",
    "str-str",
    "str-int",
    "str-float",
    "model-model",
    "model-badmodel",
    "model-int",
    "model-subclass",
    "union-type-uint32",
    "union-type-float32",
    "union-type-uint64",
    "union-type-float64",
    "union-type-str",
]

if YES_PIPE:
    DTYPE_CASES.extend(
        [
            ValidationCase(annotation=UNION_PIPE, dtype=np.uint32, passes=True),
            ValidationCase(annotation=UNION_PIPE, dtype=np.float32, passes=True),
            ValidationCase(annotation=UNION_PIPE, dtype=np.uint64, passes=False),
            ValidationCase(annotation=UNION_PIPE, dtype=np.float64, passes=False),
            ValidationCase(annotation=UNION_PIPE, dtype=str, passes=False),
        ]
    )
    DTYPE_IDS.extend(
        [
            "union-pipe-uint32",
            "union-pipe-float32",
            "union-pipe-uint64",
            "union-pipe-float64",
            "union-pipe-str",
        ]
    )
