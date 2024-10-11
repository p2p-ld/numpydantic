import sys
from typing import Union

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

RGB_UNION = (("*", "*"), ("*", "*", 3), ("*", "*", 3, 4))
UNION_TYPE: TypeAlias = Union[np.uint32, np.float32]

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
        annotation_dtype=BasicModel, dtype=BasicModel, passes=True, id="model-model"
    ),
    ValidationCase(
        annotation_dtype=BasicModel, dtype=BadModel, passes=False, id="model-badmodel"
    ),
    ValidationCase(
        annotation_dtype=BasicModel, dtype=int, passes=False, id="model-int"
    ),
    ValidationCase(
        annotation_dtype=BasicModel, dtype=SubClass, passes=True, id="model-subclass"
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.uint32,
        passes=True,
        id="union-type-uint32",
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.float32,
        passes=True,
        id="union-type-float32",
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.uint64,
        passes=False,
        id="union-type-uint64",
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE,
        dtype=np.float64,
        passes=False,
        id="union-type-float64",
    ),
    ValidationCase(
        annotation_dtype=UNION_TYPE, dtype=str, passes=False, id="union-type-str"
    ),
]
"""
Base Dtype cases
"""


if YES_PIPE:
    UNION_PIPE: TypeAlias = np.uint32 | np.float32

    DTYPE_CASES.extend(
        [
            ValidationCase(
                annotation_dtype=UNION_PIPE,
                dtype=np.uint32,
                passes=True,
                id="union-pipe-uint32",
            ),
            ValidationCase(
                annotation_dtype=UNION_PIPE,
                dtype=np.float32,
                passes=True,
                id="union-pipe-float32",
            ),
            ValidationCase(
                annotation_dtype=UNION_PIPE,
                dtype=np.uint64,
                passes=False,
                id="union-pipe-uint64",
            ),
            ValidationCase(
                annotation_dtype=UNION_PIPE,
                dtype=np.float64,
                passes=False,
                id="union-pipe-float64",
            ),
            ValidationCase(
                annotation_dtype=UNION_PIPE,
                dtype=str,
                passes=False,
                id="union-pipe-str",
            ),
        ]
    )

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
