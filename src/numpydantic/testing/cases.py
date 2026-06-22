from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from pydantic import BaseModel

from numpydantic.dtype import Float, Integer, Number
from numpydantic.testing.helpers import ValidationCase, merged_product
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


class BasicModel(BaseModel):
    x: int


class BadModel(BaseModel):
    x: int


class SubClass(BasicModel):
    pass


# --------------------------------------------------
# Annotations
# --------------------------------------------------

RGB_UNION = (("*", "*"), ("*", "*", 3), ("*", "*", 3, 4))
UNION_TYPE: TypeAlias = np.uint32 | np.float32

SHAPE_CASES = (
    ValidationCase(shape=(10, 10, 2, 2), passes=True, id="valid shape"),
    ValidationCase(shape=(10, 10, 2), passes=False, id="missing dimension"),
    ValidationCase(shape=(10, 10, 2, 2, 2), passes=False, id="extra dimension"),
    ValidationCase(shape=(11, 10, 2, 2), passes=False, id="dimension too large"),
    ValidationCase(shape=(9, 10, 2, 2), passes=False, id="dimension too small"),
    ValidationCase(shape=(10, 10, 1, 1), passes=True, id="wildcard smaller"),
    ValidationCase(shape=(10, 10, 3, 3), passes=True, id="wildcard larger"),
    ValidationCase(
        annotation_shape=RGB_UNION, shape=(5, 5), passes=True, id="Union 2D"
    ),
    ValidationCase(
        annotation_shape=RGB_UNION, shape=(5, 5, 3), passes=True, id="Union 3D"
    ),
    ValidationCase(
        annotation_shape=RGB_UNION, shape=(5, 5, 3, 4), passes=True, id="Union 4D"
    ),
    ValidationCase(
        annotation_shape=RGB_UNION,
        shape=(5, 5, 4),
        passes=False,
        id="Union incorrect 3D",
    ),
    ValidationCase(
        annotation_shape=RGB_UNION,
        shape=(5, 5, 3, 6),
        passes=False,
        id="Union incorrect 4D",
    ),
    ValidationCase(
        annotation_shape=RGB_UNION,
        shape=(5, 5, 4, 6),
        passes=False,
        id="Union incorrect both",
    ),
    ValidationCase(
        annotation_shape=None,
        shape=tuple(),
        id="scalar",
        marks={"scalar"},
        passes=True,
    ),
    ValidationCase(
        annotation_shape=("*", "*"),
        shape=tuple(),
        id="scalar-min-dimensions",
        marks={"scalar"},
        passes=False,
    ),
    ValidationCase(
        annotation_shape=("*", 3), shape=(0, 3), id="zero-length", passes=True
    ),
)
"""
Base Shape cases
"""


DTYPE_CASES = [
    ValidationCase(dtype=float, passes=True, id="float"),
    ValidationCase(dtype=int, passes=False, id="int"),
    ValidationCase(dtype=np.uint8, passes=False, id="uint8"),
    ValidationCase(annotation_dtype=Number, dtype=int, passes=True, id="number-int"),
    ValidationCase(
        annotation_dtype=Number, dtype=float, passes=True, id="number-float"
    ),
    ValidationCase(
        annotation_dtype=Number, dtype=np.uint8, passes=True, id="number-uint8"
    ),
    ValidationCase(
        annotation_dtype=Number, dtype=np.float16, passes=True, id="number-float16"
    ),
    ValidationCase(annotation_dtype=Number, dtype=str, passes=False, id="number-str"),
    ValidationCase(annotation_dtype=Integer, dtype=int, passes=True, id="integer-int"),
    ValidationCase(
        annotation_dtype=Integer, dtype=np.uint8, passes=True, id="integer-uint8"
    ),
    ValidationCase(
        annotation_dtype=Integer, dtype=float, passes=False, id="integer-float"
    ),
    ValidationCase(
        annotation_dtype=Integer, dtype=np.float32, passes=False, id="integer-float32"
    ),
    ValidationCase(annotation_dtype=Integer, dtype=str, passes=False, id="integer-str"),
    ValidationCase(annotation_dtype=Float, dtype=float, passes=True, id="float-float"),
    ValidationCase(
        annotation_dtype=Float, dtype=np.float32, passes=True, id="float-float32"
    ),
    ValidationCase(annotation_dtype=Float, dtype=int, passes=False, id="float-int"),
    ValidationCase(
        annotation_dtype=Float, dtype=np.uint8, passes=False, id="float-uint8"
    ),
    ValidationCase(annotation_dtype=Float, dtype=str, passes=False, id="float-str"),
    ValidationCase(annotation_dtype=str, dtype=str, passes=True, id="str-str"),
    ValidationCase(annotation_dtype=str, dtype=int, passes=False, id="str-int"),
    ValidationCase(annotation_dtype=str, dtype=float, passes=False, id="str-float"),
    ValidationCase(
        annotation_dtype=np.str_,
        dtype=str,
        passes=True,
        id="np_str-str",
        marks={"np_str", "str"},
    ),
    ValidationCase(
        annotation_dtype=np.str_,
        dtype=np.str_,
        passes=True,
        id="np_str-np_str",
        marks={"np_str", "str"},
    ),
    ValidationCase(
        annotation_dtype=(int, np.str_),
        dtype=str,
        passes=True,
        id="tuple_np_str-str",
        marks={"np_str", "str", "tuple"},
    ),
    ValidationCase(
        annotation_dtype=BasicModel,
        dtype=BasicModel,
        passes=True,
        id="model-model",
        marks={"model"},
    ),
    ValidationCase(
        annotation_dtype=BasicModel,
        dtype=BadModel,
        passes=False,
        id="model-badmodel",
        marks={"model"},
    ),
    ValidationCase(
        annotation_dtype=BasicModel,
        dtype=int,
        passes=False,
        id="model-int",
        marks={"model"},
    ),
    ValidationCase(
        annotation_dtype=BasicModel,
        dtype=SubClass,
        passes=True,
        id="model-subclass",
        marks={"model"},
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.uint32,
        passes=True,
        id="union-type-uint32",
        marks={"union"},
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.float32,
        passes=True,
        id="union-type-float32",
        marks={"union"},
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.uint64,
        passes=False,
        id="union-type-uint64",
        marks={"union"},
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.float64,
        passes=False,
        id="union-type-float64",
        marks={"union"},
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=str,
        passes=False,
        id="union-type-str",
        marks={"union"},
    ),
    ValidationCase(
        annotation_dtype=datetime,
        dtype=datetime,
        passes=True,
        id="datetime-datetime",
        marks={"datetime"},
    ),
    ValidationCase(
        annotation_dtype=datetime,
        dtype=np.datetime64,
        passes=True,
        id="datetime-datetime64",
        marks={"datetime"},
    ),
    ValidationCase(
        annotation_dtype=np.datetime64,
        dtype=datetime,
        passes=False,
        id="datetime64-datetime",
        marks={"datetime"},
    ),
    ValidationCase(
        annotation_dtype=np.datetime64,
        dtype=np.datetime64,
        passes=True,
        id="datetime64-datetime64",
        marks={"datetime"},
    ),
]
"""
Base Dtype cases
"""

INTERFACE_CASES = [
    ValidationCase(interface=NumpyCase, id="numpy"),
    ValidationCase(interface=HDF5Case, id="hdf5"),
    ValidationCase(interface=HDF5CompoundCase, id="hdf5_compound"),
    ValidationCase(interface=DaskCase, id="dask"),
    ValidationCase(interface=ZarrCase, id="zarr"),
    ValidationCase(interface=ZarrDirCase, id="zarr_dir"),
    ValidationCase(interface=ZarrZipCase, id="zarr_zip"),
    ValidationCase(interface=ZarrNestedCase, id="zarr_nested"),
    ValidationCase(interface=VideoCase, id="video"),
]
"""
All the interface cases
"""


DTYPE_AND_SHAPE_CASES = merged_product(SHAPE_CASES, DTYPE_CASES)
"""
Merged product of dtype and shape cases
"""
DTYPE_AND_SHAPE_CASES_PASSING = merged_product(
    SHAPE_CASES, DTYPE_CASES, conditions={"passes": True}
)
"""
Merged product of dtype and shape cases that are valid
"""

DTYPE_AND_INTERFACE_CASES = merged_product(INTERFACE_CASES, DTYPE_CASES)
"""
Merged product of dtype and interface cases
"""
DTYPE_AND_INTERFACE_CASES_PASSING = merged_product(
    INTERFACE_CASES, DTYPE_CASES, conditions={"passes": True}
)
"""
Merged product of dtype and interface cases that pass
"""

ALL_CASES = merged_product(SHAPE_CASES, DTYPE_CASES, INTERFACE_CASES)
"""
Merged product of all cases - dtype, shape, and interface
"""
ALL_CASES_PASSING = merged_product(
    SHAPE_CASES, DTYPE_CASES, INTERFACE_CASES, conditions={"passes": True}
)
"""
Merged product of all cases, but only those that pass
"""

ZERO_LENGTH_CASES_PASSING = [c for c in ALL_CASES_PASSING if "zero-length" in c.id]


# ---------------------------------------------------------------------------
# Mypy cases
# ---------------------------------------------------------------------------
#
# Slight repetition, but more focused than the above cases -
# we only want to check the general typing forms, rather than be exhaustive about
# all combinations of shapes and dtypes, as we do with actual runtime tests.

MYPY_SHAPE_CASES = [
    ValidationCase(
        id="mypy-literal", annotation_shape=(3, 3), shape=(3, 3), passes=True
    ),
    ValidationCase(
        id="mypy-literal-badshape", annotation_shape=(3, 3), shape=(2, 4), passes=False
    ),
    ValidationCase(
        id="mypy-literal-badcardinality",
        annotation_shape=(3, 3),
        shape=(3, 3, 3),
        passes=False,
    ),
    ValidationCase(
        id="mypy-wildcard", annotation_shape=(3, "*"), shape=(3, 5), passes=True
    ),
    ValidationCase(
        id="mypy-wildcard-badshape",
        annotation_shape=(3, "*"),
        shape=(2, 5),
        passes=False,
    ),
    ValidationCase(
        id="mypy-wildcard-badcardinality",
        annotation_shape=(3, "*"),
        shape=(3, 3, 3),
        passes=False,
    ),
    ValidationCase(
        id="mypy-ellipsis",
        annotation_shape=(3, 3, "..."),
        shape=(3, 3, 3, 3),
        passes=True,
    ),
    ValidationCase(
        id="mypy-ellipsis-samecardinality",
        annotation_shape=(3, 3, "..."),
        shape=(3, 3),
        passes=True,
    ),
    ValidationCase(
        id="mypy-ellipsis-badshape",
        annotation_shape=(3, 3, "..."),
        shape=(2, 2),
        passes=False,
    ),
    ValidationCase(
        id="mypy-range",
        annotation_shape=("2-5", "2-5", "2-*"),
        shape=(2, 5, 10),
        passes=True,
    ),
    ValidationCase(
        id="mypy-range-badshape",
        annotation_shape=("2-5", "2-5", "2-*"),
        shape=(1, 6, 10),
        passes=False,
    ),
    ValidationCase(
        id="mypy-range-badwildcard",
        annotation_shape=("2-5", "2-5", "2-*"),
        shape=(2, 5, 1),
        passes=False,
    ),
]

MYPY_DTYPE_CASES = [
    ValidationCase(
        id="mypy-uint8", annotation_dtype=np.uint8, dtype=np.uint8, passes=True
    ),
    ValidationCase(
        id="mypy-uint8-baddtype",
        annotation_dtype=np.uint8,
        dtype=np.float64,
        passes=False,
    ),
    ValidationCase(
        id="mypy-uint8-builtin", annotation_dtype=np.uint8, dtype=int, passes=False
    ),
    ValidationCase(id="mypy-builtin", annotation_dtype=float, dtype=float, passes=True),
    ValidationCase(
        id="mypy-builtin-float64", annotation_dtype=float, dtype=np.float64, passes=True
    ),
    ValidationCase(
        id="mypy-builtin-float64", annotation_dtype=float, dtype=int, passes=False
    ),
    ValidationCase(id="mypy-any", annotation_dtype=Any, dtype=np.float64, passes=True),
    ValidationCase(
        id="mypy-union-int32",
        annotation_dtype=np.uint32 | np.float32,
        dtype=np.uint32,
        passes=True,
    ),
    ValidationCase(
        id="mypy-union-float32",
        annotation_dtype=np.uint32 | np.float32,
        dtype=np.float32,
        passes=True,
    ),
    ValidationCase(
        id="mypy-union-uint8",
        annotation_dtype=np.uint32 | np.float32,
        dtype=np.uint8,
        passes=False,
    ),
    ValidationCase(
        id="mypy-tuple", annotation_dtype="Integer", dtype=np.uint8, passes=True
    ),
    ValidationCase(
        id="mypy-tuple-baddtype",
        annotation_dtype="Integer",
        dtype=np.float64,
        passes=False,
    ),
]

MYPY_CASES = merged_product(
    MYPY_SHAPE_CASES + MYPY_DTYPE_CASES,
    [c for c in INTERFACE_CASES if c.interface in (NumpyCase, DaskCase, ZarrCase)],
)
