import pytest

import dask.array as da
from pydantic import ValidationError

from numpydantic.interface import DaskInterface
from numpydantic.exceptions import DtypeError, ShapeError

from tests.conftest import ValidationCase


def dask_array(case: ValidationCase) -> da.Array:
    return da.zeros(shape=case.shape, dtype=case.dtype, chunks=10)


def _test_dask_case(case: ValidationCase):
    array = dask_array(case)
    if case.passes:
        case.model(array=array)
    else:
        with pytest.raises((ValidationError, DtypeError, ShapeError)):
            case.model(array=array)


def test_dask_enabled():
    """
    We need dask to be available to run these tests :)
    """
    assert DaskInterface.enabled()


def test_dask_check(interface_type):
    if interface_type[1] is DaskInterface:
        assert DaskInterface.check(interface_type[0])
    else:
        assert not DaskInterface.check(interface_type[0])


def test_dask_shape(shape_cases):
    _test_dask_case(shape_cases)


def test_dask_dtype(dtype_cases):
    _test_dask_case(dtype_cases)
