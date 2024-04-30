import pytest

import numpy as np
import dask.array as da
import zarr

from numpydantic import interface
from tests.fixtures import hdf5_array, zarr_nested_array, zarr_array


@pytest.fixture(
    scope="function",
    params=[
        ([[1, 2], [3, 4]], interface.NumpyInterface),
        (np.zeros((3, 4)), interface.NumpyInterface),
        (hdf5_array, interface.H5Interface),
        (da.random.random((10, 10)), interface.DaskInterface),
        (zarr.ones((10, 10)), interface.ZarrInterface),
        (zarr_nested_array, interface.ZarrInterface),
        (zarr_array, interface.ZarrInterface),
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
    return request.param
