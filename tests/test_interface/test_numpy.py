import numpy as np
import pytest

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


def test_numpy_coercion(model_blank):
    """If no other interface matches, we try and coerce to a numpy array"""
    instance = model_blank(array=[1, 2, 3])
    assert isinstance(instance.array, np.ndarray)
