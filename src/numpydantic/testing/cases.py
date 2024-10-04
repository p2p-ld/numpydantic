import sys
from collections.abc import Sequence
from itertools import product
from typing import Generator, Union

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.dtype import Float, Integer, Number
from numpydantic.testing.helpers import ValidationCase, merge_cases
from numpydantic.testing.interfaces import (
    DaskCase,
    HDF5Case,
    HDF5CompoundCase,
    NumpyCase,
    VideoCase,
    ZarrCase,
    ZarrDirCase,
    ZarrNestedCase,
    ZarrZipCase,
)

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


# --------------------------------------------------
# Annotations
# --------------------------------------------------

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

SHAPE_CASES = (
    ValidationCase(shape=(10, 10, 10), passes=True, id="valid shape"),
    ValidationCase(shape=(10, 10), passes=False, id="missing dimension"),
    ValidationCase(shape=(10, 10, 10, 10), passes=False, id="extra dimension"),
    ValidationCase(shape=(11, 10, 10), passes=False, id="dimension too large"),
    ValidationCase(shape=(9, 10, 10), passes=False, id="dimension too small"),
    ValidationCase(shape=(10, 10, 9), passes=True, id="wildcard smaller"),
    ValidationCase(shape=(10, 10, 11), passes=True, id="wildcard larger"),
    ValidationCase(annotation=RGB_UNION, shape=(5, 5), passes=True, id="Union 2D"),
    ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3), passes=True, id="Union 3D"),
    ValidationCase(
        annotation=RGB_UNION, shape=(5, 5, 3, 4), passes=True, id="Union 4D"
    ),
    ValidationCase(
        annotation=RGB_UNION, shape=(5, 5, 4), passes=False, id="Union incorrect 3D"
    ),
    ValidationCase(
        annotation=RGB_UNION, shape=(5, 5, 3, 6), passes=False, id="Union incorrect 4D"
    ),
    ValidationCase(
        annotation=RGB_UNION,
        shape=(5, 5, 4, 6),
        passes=False,
        id="Union incorrect both",
    ),
)


DTYPE_CASES = [
    ValidationCase(dtype=float, passes=True, id="float"),
    ValidationCase(dtype=int, passes=False, id="int"),
    ValidationCase(dtype=np.uint8, passes=False, id="uint8"),
    ValidationCase(annotation=NUMBER, dtype=int, passes=True, id="number-int"),
    ValidationCase(annotation=NUMBER, dtype=float, passes=True, id="number-float"),
    ValidationCase(annotation=NUMBER, dtype=np.uint8, passes=True, id="number-uint8"),
    ValidationCase(
        annotation=NUMBER, dtype=np.float16, passes=True, id="number-float16"
    ),
    ValidationCase(annotation=NUMBER, dtype=str, passes=False, id="number-str"),
    ValidationCase(annotation=INTEGER, dtype=int, passes=True, id="integer-int"),
    ValidationCase(annotation=INTEGER, dtype=np.uint8, passes=True, id="integer-uint8"),
    ValidationCase(annotation=INTEGER, dtype=float, passes=False, id="integer-float"),
    ValidationCase(
        annotation=INTEGER, dtype=np.float32, passes=False, id="integer-float32"
    ),
    ValidationCase(annotation=INTEGER, dtype=str, passes=False, id="integer-str"),
    ValidationCase(annotation=FLOAT, dtype=float, passes=True, id="float-float"),
    ValidationCase(annotation=FLOAT, dtype=np.float32, passes=True, id="float-float32"),
    ValidationCase(annotation=FLOAT, dtype=int, passes=False, id="float-int"),
    ValidationCase(annotation=FLOAT, dtype=np.uint8, passes=False, id="float-uint8"),
    ValidationCase(annotation=FLOAT, dtype=str, passes=False, id="float-str"),
    ValidationCase(annotation=STRING, dtype=str, passes=True, id="str-str"),
    ValidationCase(annotation=STRING, dtype=int, passes=False, id="str-int"),
    ValidationCase(annotation=STRING, dtype=float, passes=False, id="str-float"),
    ValidationCase(annotation=MODEL, dtype=BasicModel, passes=True, id="model-model"),
    ValidationCase(annotation=MODEL, dtype=BadModel, passes=False, id="model-badmodel"),
    ValidationCase(annotation=MODEL, dtype=int, passes=False, id="model-int"),
    ValidationCase(annotation=MODEL, dtype=SubClass, passes=True, id="model-subclass"),
    ValidationCase(
        annotation=UNION_TYPE, dtype=np.uint32, passes=True, id="union-type-uint32"
    ),
    ValidationCase(
        annotation=UNION_TYPE, dtype=np.float32, passes=True, id="union-type-float32"
    ),
    ValidationCase(
        annotation=UNION_TYPE, dtype=np.uint64, passes=False, id="union-type-uint64"
    ),
    ValidationCase(
        annotation=UNION_TYPE, dtype=np.float64, passes=False, id="union-type-float64"
    ),
    ValidationCase(annotation=UNION_TYPE, dtype=str, passes=False, id="union-type-str"),
]


if YES_PIPE:
    DTYPE_CASES.extend(
        [
            ValidationCase(
                annotation=UNION_PIPE,
                dtype=np.uint32,
                passes=True,
                id="union-pipe-uint32",
            ),
            ValidationCase(
                annotation=UNION_PIPE,
                dtype=np.float32,
                passes=True,
                id="union-pipe-float32",
            ),
            ValidationCase(
                annotation=UNION_PIPE,
                dtype=np.uint64,
                passes=False,
                id="union-pipe-uint64",
            ),
            ValidationCase(
                annotation=UNION_PIPE,
                dtype=np.float64,
                passes=False,
                id="union-pipe-float64",
            ),
            ValidationCase(
                annotation=UNION_PIPE, dtype=str, passes=False, id="union-pipe-str"
            ),
        ]
    )

_INTERFACE_CASES = [
    NumpyCase,
    HDF5Case,
    HDF5CompoundCase,
    DaskCase,
    ZarrCase,
    ZarrDirCase,
    ZarrZipCase,
    ZarrNestedCase,
    VideoCase,
]


def merged_product(
    *args: Sequence[ValidationCase],
) -> Generator[ValidationCase, None, None]:
    """
    Generator for the product of the iterators of validation cases,
    merging each tuple, and respecting if they should be :meth:`.ValidationCase.skip`
    or not.

    Examples:

        .. code-block:: python

            shape_cases = [
                ValidationCase(shape=(10, 10, 10), passes=True, id="valid shape"),
                ValidationCase(shape=(10, 10), passes=False, id="missing dimension"),
            ]
            dtype_cases = [
                ValidationCase(dtype=float, passes=True, id="float"),
                ValidationCase(dtype=int, passes=False, id="int"),
            ]

            iterator = merged_product(shape_cases, dtype_cases))
            next(iterator)
            # ValidationCase(shape=(10, 10, 10), dtype=float, passes=True, id="valid shape-float")
            next(iterator)
            # ValidationCase(shape=(10, 10, 10), dtype=int, passes=False, id="valid shape-int")


    """
    iterator = product(*args)
    for case_tuple in iterator:
        case = merge_cases(case_tuple)
        if case.skip():
            continue
        yield case
