import pytest

from typing import Callable, Tuple, Type
import numpy as np
import dask.array as da
import zarr
from pydantic import BaseModel

from numpydantic import interface, NDArray


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            ([[1, 2], [3, 4]], interface.NumpyInterface),
            marks=pytest.mark.numpy,
            id="numpy-list",
        ),
        pytest.param(
            (np.zeros((3, 4)), interface.NumpyInterface),
            marks=pytest.mark.numpy,
            id="numpy",
        ),
        pytest.param(
            ("hdf5_array", interface.H5Interface),
            marks=pytest.mark.hdf5,
            id="h5-array-path",
        ),
        pytest.param(
            (da.random.random((10, 10)), interface.DaskInterface),
            marks=pytest.mark.dask,
            id="dask",
        ),
        pytest.param(
            (zarr.ones((10, 10)), interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-memory",
        ),
        pytest.param(
            ("zarr_nested_array", interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-nested",
        ),
        pytest.param(
            ("zarr_array", interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-array",
        ),
        pytest.param(
            ("avi_video", interface.VideoInterface), marks=pytest.mark.video, id="video"
        ),
    ],
)
def interface_type(request) -> Tuple[NDArray, Type[interface.Interface]]:
    """
    Test cases for each interface's ``check`` method - each input should match the
    provided interface and that interface only
    """
    if isinstance(request.param[0], str):
        return (request.getfixturevalue(request.param[0]), request.param[1])
    else:
        return request.param


@pytest.fixture()
def all_interfaces(interface_type) -> BaseModel:
    """
    An instantiated version of each interface within a basemodel,
    with the array in an `array` field
    """
    array, interface = interface_type
    if isinstance(array, Callable):
        array = array()

    class MyModel(BaseModel):
        array: NDArray

    instance = MyModel(array=array)
    return instance
