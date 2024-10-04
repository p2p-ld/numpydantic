import inspect
from typing import Callable, Tuple, Type

import pytest
from pydantic import BaseModel

from numpydantic import NDArray, interface
from numpydantic.testing.helpers import InterfaceCase
from numpydantic.testing.interfaces import (
    DaskCase,
    HDF5Case,
    NumpyCase,
    VideoCase,
    ZarrCase,
    ZarrDirCase,
    ZarrNestedCase,
)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            ([[1, 2], [3, 4]], interface.NumpyInterface),
            marks=pytest.mark.numpy,
            id="numpy-list",
        ),
        pytest.param(
            (NumpyCase, interface.NumpyInterface),
            marks=pytest.mark.numpy,
            id="numpy",
        ),
        pytest.param(
            (HDF5Case, interface.H5Interface),
            marks=pytest.mark.hdf5,
            id="h5-array-path",
        ),
        pytest.param(
            (DaskCase, interface.DaskInterface),
            marks=pytest.mark.dask,
            id="dask",
        ),
        pytest.param(
            (ZarrCase, interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-memory",
        ),
        pytest.param(
            (ZarrNestedCase, interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-nested",
        ),
        pytest.param(
            (ZarrDirCase, interface.ZarrInterface),
            marks=pytest.mark.zarr,
            id="zarr-dir",
        ),
        pytest.param(
            (VideoCase, interface.VideoInterface), marks=pytest.mark.video, id="video"
        ),
    ],
)
def interface_type(
    request, tmp_output_dir_func
) -> Tuple[NDArray, Type[interface.Interface]]:
    """
    Test cases for each interface's ``check`` method - each input should match the
    provided interface and that interface only
    """

    if inspect.isclass(request.param[0]) and issubclass(
        request.param[0], InterfaceCase
    ):
        array = request.param[0].make_array(path=tmp_output_dir_func)
        if array is None:
            pytest.skip()
        return array, request.param[1]
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
