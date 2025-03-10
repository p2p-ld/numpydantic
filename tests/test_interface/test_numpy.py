from datetime import datetime

import numpy as np
import pytest
from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.testing.cases import NumpyCase

pytestmark = pytest.mark.numpy


@pytest.mark.shape
def test_numpy_shape(shape_cases):
    shape_cases.interface = NumpyCase
    shape_cases.validate_case()


@pytest.mark.dtype
def test_numpy_dtype(dtype_cases):
    dtype_cases.interface = NumpyCase
    dtype_cases.validate_case()


@pytest.mark.dtype
def test_numpy_datetime64():
    """
    Special testing for datetime64, since it's a numpy-specific thing,
    and requires special handling in the other interfaces.

    Typically, we expect that generic type annotation to be a `datetime.datetime`,
    but it should also be possible to use a numpy.datetime64 type when doing
    numpy validation.

    So this test is just the simplest regression test to make sure we can use datetime64
    """
    arr = np.array([np.datetime64(datetime.now())], dtype=np.datetime64)

    annotation = NDArray[Shape["*"], np.datetime64]
    _ = annotation(arr)

    class MyModel(BaseModel):
        array: annotation

    _ = MyModel(array=arr)


def test_numpy_coercion(model_blank):
    """If no other interface matches, we try and coerce to a numpy array"""
    instance = model_blank(array=[1, 2, 3])
    assert isinstance(instance.array, np.ndarray)
