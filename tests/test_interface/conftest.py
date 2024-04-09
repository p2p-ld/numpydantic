import pytest

import numpy as np
import dask.array as da

from numpydantic import interface
from tests.conftest import h5_array, h5file


@pytest.fixture(
    scope="function",
    params=[
        ([[1, 2], [3, 4]], interface.NumpyInterface),
        (np.zeros((3, 4)), interface.NumpyInterface),
        (h5_array, interface.H5Interface),
        (da.random.random((10, 10)), interface.DaskInterface),
    ],
    ids=["numpy_list", "numpy", "H5Array", "dask"],
)
def interface_type(request):
    return request.param
