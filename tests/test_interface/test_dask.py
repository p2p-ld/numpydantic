import json

import dask.array as da
import pytest

from numpydantic.interface import DaskInterface
from numpydantic.testing.interfaces import DaskCase

pytestmark = pytest.mark.dask


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


@pytest.mark.shape
def test_dask_shape(shape_cases):
    shape_cases.interface = DaskCase
    shape_cases.validate_case()


@pytest.mark.dtype
def test_dask_dtype(dtype_cases):
    dtype_cases.interface = DaskCase
    dtype_cases.validate_case()


@pytest.mark.serialization
def test_dask_to_json(array_model):
    array_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    array = da.array(array_list)
    model = array_model((3, 3), int)
    instance = model(array=array)
    jsonified = json.loads(instance.model_dump_json())
    assert jsonified["array"] == array_list
