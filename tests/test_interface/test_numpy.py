import json
from datetime import datetime
from typing import Any

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


@pytest.mark.scalar
@pytest.mark.serialization
def test_numpy_empty_string():
    """
    Empty strings are coerced to arrays and serialzied as such.

    Mildly redundant with test_serialize_scalars_as_arrays below,
    but a specific regression test for issue #52

    References:
        - https://github.com/p2p-ld/numpydantic/issues/52
    """

    class MyModel(BaseModel):
        array: NDArray[Any, np.str_]

    inst = MyModel(array="")
    assert isinstance(inst.array, np.ndarray)
    assert inst.array == np.array([""])
    dumped = inst.model_dump_json()
    assert json.loads(dumped)["array"] == [""]


@pytest.mark.serialization
@pytest.mark.scalar
@pytest.mark.parametrize("scalar", ("", 0, 0.5))
@pytest.mark.parametrize("as_json", (True, False))
def test_serialize_scalars_as_arrays(scalar, as_json: bool):
    """
    The numpy interface matches scalar values, coerces them to arrays,
    and serializes them as arrays in both python and json.
    """

    class MyModel(BaseModel):
        array: NDArray

    inst = MyModel(array=scalar)
    assert isinstance(inst.array, np.ndarray)
    assert inst.array == np.array([scalar])
    if as_json:
        dumped = inst.model_dump_json()
        assert json.loads(dumped)["array"] == [scalar]
    else:
        dumped = inst.model_dump()
        assert dumped["array"] == np.array([scalar])
