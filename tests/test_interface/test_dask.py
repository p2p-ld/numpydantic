import pytest

import dask.array as da
from pydantic import ValidationError

from numpydantic.interface import DaskInterface


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


@pytest.mark.parametrize(
    "array,passes",
    [
        (da.random.random((5, 10)), True),
        (da.random.random((5, 10, 3)), True),
        (da.random.random((5, 10, 3, 4)), True),
        (da.random.random((5, 10, 4)), False),
        (da.random.random((5, 10, 3, 6)), False),
        (da.random.random((5, 10, 4, 6)), False),
    ],
)
def test_dask_shape(model_rgb, array, passes):
    if passes:
        model_rgb(array=array)
    else:
        with pytest.raises(ValidationError):
            model_rgb(array=array)


@pytest.mark.skip("TODO")
def test_dask_dtype():
    pass
