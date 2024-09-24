import sys

import pytest
from typing import Any, Tuple, Union, Type

from pydantic import BaseModel, computed_field, ConfigDict
from numpydantic import NDArray, Shape
from numpydantic.ndarray import NDArrayMeta
from numpydantic.dtype import Float, Number, Integer
import numpy as np

from tests.fixtures import *

if sys.version_info.minor >= 10:
    from typing import TypeAlias

    YES_PIPE = True
else:
    from typing_extensions import TypeAlias

    YES_PIPE = False


def pytest_addoption(parser):
    parser.addoption(
        "--with-output",
        action="store_true",
        help="Keep test outputs in the __tmp__ directory",
    )


class ValidationCase(BaseModel):
    """
    Test case for validating an array.

    Contains both the validating model and the parameterization for an array to
    test in a given interface
    """

    annotation: Any = NDArray[Shape["10, 10, *"], Float]
    """
    Array annotation used in the validating model
    Any typed because the types of type annotations are weird
    """
    shape: Tuple[int, ...] = (10, 10, 10)
    """Shape of the array to validate"""
    dtype: Union[Type, np.dtype] = float
    """Dtype of the array to validate"""
    passes: bool
    """Whether the validation should pass or not"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field()
    def model(self) -> Type[BaseModel]:
        """A model with a field ``array`` with the given annotation"""
        annotation = self.annotation

        class Model(BaseModel):
            array: annotation

        return Model


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
if YES_PIPE:
    UNION_PIPE: TypeAlias = NDArray[Shape["*, *, *"], np.uint32 | np.float32]


@pytest.fixture(
    scope="module",
    params=[
        ValidationCase(shape=(10, 10, 10), passes=True),
        ValidationCase(shape=(10, 10), passes=False),
        ValidationCase(shape=(10, 10, 10, 10), passes=False),
        ValidationCase(shape=(11, 10, 10), passes=False),
        ValidationCase(shape=(9, 10, 10), passes=False),
        ValidationCase(shape=(10, 10, 9), passes=True),
        ValidationCase(shape=(10, 10, 11), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3, 4), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 4), passes=False),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3, 6), passes=False),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 4, 6), passes=False),
    ],
    ids=[
        "valid shape",
        "missing dimension",
        "extra dimension",
        "dimension too large",
        "dimension too small",
        "wildcard smaller",
        "wildcard larger",
        "Union 2D",
        "Union 3D",
        "Union 4D",
        "Union incorrect 3D",
        "Union incorrect 4D",
        "Union incorrect both",
    ],
)
def shape_cases(request) -> ValidationCase:
    return request.param


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


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=DTYPE_IDS)
def dtype_cases(request) -> ValidationCase:
    return request.param
