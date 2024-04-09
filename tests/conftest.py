import pytest
from pathlib import Path
from typing import Optional, Union, Type

import h5py
import numpy as np
from pydantic import BaseModel, Field

from numpydantic.interface.hdf5 import H5Array
from numpydantic import NDArray, Shape
from nptyping import Number


@pytest.fixture(scope="session")
def model_rgb() -> Type[BaseModel]:
    class RGB(BaseModel):
        array: Optional[
            Union[
                NDArray[Shape["* x, * y"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
            ]
        ] = Field(None)

    return RGB


@pytest.fixture(scope="function")
def h5file(tmp_path) -> h5py.File:
    h5f = h5py.File(tmp_path / "file.h5", "w")
    yield h5f
    h5f.close()


@pytest.fixture(scope="function")
def h5_array(h5file) -> H5Array:
    """trivial hdf5 array used for testing array existence"""
    path = "/data"
    h5file.create_dataset(path, data=np.zeros((3, 4)))
    return H5Array(file=Path(h5file.filename), path=path)
