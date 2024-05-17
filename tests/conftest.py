import pdb
import sys

import pytest
from typing import Any, Tuple, Union, Type

if sys.version_info.minor >= 10:
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from pydantic import BaseModel, computed_field, ConfigDict
from numpydantic import NDArray, Shape
from numpydantic.ndarray import NDArrayMeta
from numpydantic.dtype import Float, Number, Integer
import numpy as np

from tests.fixtures import *


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


RGB_UNION: TypeAlias = Union[
    NDArray[Shape["* x, * y"], Number],
    NDArray[Shape["* x, * y, 3 r_g_b"], Number],
    NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
]

NUMBER: TypeAlias = NDArray[Shape["*, *, *"], Number]
INTEGER: TypeAlias = NDArray[Shape["*, *, *"], Integer]
FLOAT: TypeAlias = NDArray[Shape["*, *, *"], Float]


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


@pytest.fixture(
    scope="module",
    params=[
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
        ValidationCase(annotation=FLOAT, dtype=float, passes=True),
        ValidationCase(annotation=FLOAT, dtype=np.float32, passes=True),
        ValidationCase(annotation=FLOAT, dtype=int, passes=False),
        ValidationCase(annotation=FLOAT, dtype=np.uint8, passes=False),
    ],
    ids=[
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
        "float-float",
        "float-float32",
        "float-int",
        "float-uint8",
    ],
)
def dtype_cases(request) -> ValidationCase:
    return request.param
