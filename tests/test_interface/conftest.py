import pytest

import numpy as np
import dask.array as da
import zarr

from numpydantic import interface


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
    ],
    ids=[
        "numpy_list",
        "numpy",
        "H5ArrayPath",
        "dask",
        "zarr_memory",
        "zarr_nested",
        "zarr_array",
    ],
)
def interface_type(request):
    """
    Test cases for each interface's ``check`` method - each input should match the
    provided interface and that interface only
    """
    if isinstance(request.param[0], str):
        return (request.getfixturevalue(request.param[0]), request.param[1])
    else:
        return request.param
