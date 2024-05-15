import numpy as np
import pytest
from pydantic import ValidationError
from numpydantic.exceptions import DtypeError, ShapeError

from tests.conftest import ValidationCase


def numpy_array(case: ValidationCase) -> np.ndarray:
    return np.zeros(shape=case.shape, dtype=case.dtype)


def _test_np_case(case: ValidationCase):
    array = numpy_array(case)
    if case.passes:
        case.model(array=array)
    else:
        with pytest.raises((ValidationError, DtypeError, ShapeError)):
            case.model(array=array)


def test_numpy_shape(shape_cases):
    _test_np_case(shape_cases)


def test_numpy_dtype(dtype_cases):
    _test_np_case(dtype_cases)


def test_numpy_coercion(model_blank):
    """If no other interface matches, we try and coerce to a numpy array"""
    instance = model_blank(array=[1, 2, 3])
    assert isinstance(instance.array, np.ndarray)
