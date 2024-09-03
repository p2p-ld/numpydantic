import pytest

from typing import Tuple, Callable
import numpy as np
import dask.array as da
import zarr
from pydantic import BaseModel

from numpydantic import interface, NDArray


@pytest.fixture(
    scope="function",
    params=[
        ([[1, 2], [3, 4]], interface.NumpyInterface),
        (np.zeros((3, 4)), interface.NumpyInterface),
        ("hdf5_array", interface.H5Interface),
        (da.random.random((10, 10)), interface.DaskInterface),
        (zarr.ones((10, 10)), interface.ZarrInterface),
        ("zarr_nested_array", interface.ZarrInterface),
        ("zarr_array", interface.ZarrInterface),
        ("avi_video", interface.VideoInterface),
    ],
    ids=[
        "numpy_list",
        "numpy",
        "H5ArrayPath",
        "dask",
        "zarr_memory",
        "zarr_nested",
        "zarr_array",
        "video",
    ],
)
def interface_type(request) -> Tuple[NDArray, interface.Interface]:
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
