from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pytest
import zarr

from numpydantic.interface.hdf5 import H5ArrayPath
from numpydantic.interface.zarr import ZarrArrayPath
from numpydantic.testing.interfaces import HDF5Case, HDF5CompoundCase, VideoCase


@pytest.fixture(scope="function")
def hdf5_array(
    request, tmp_output_dir_func
) -> Callable[[Tuple[int, ...], Union[np.dtype, type]], H5ArrayPath]:

    def _hdf5_array(
        shape: Tuple[int, ...] = (10, 10),
        dtype: Union[np.dtype, type] = float,
        compound: bool = False,
    ) -> H5ArrayPath:
        if compound:
            array: H5ArrayPath = HDF5CompoundCase.make_array(
                shape, dtype, tmp_output_dir_func
            )
            return array
        else:
            return HDF5Case.make_array(shape, dtype, tmp_output_dir_func)

    return _hdf5_array


@pytest.fixture(scope="function")
def zarr_nested_array(tmp_output_dir_func) -> ZarrArrayPath:
    """Zarr array within a nested array"""
    file = tmp_output_dir_func / "nested.zarr"
    path = "a/b/c"
    root = zarr.open(str(file), mode="w")
    _ = root.zeros(path, shape=(100, 100), chunks=(10, 10))
    return ZarrArrayPath(file=file, path=path)


@pytest.fixture(scope="function")
def zarr_array(tmp_output_dir_func) -> Path:
    file = tmp_output_dir_func / "array.zarr"
    array = zarr.open(str(file), mode="w", shape=(100, 100), chunks=(10, 10))
    array[:] = 0
    return file


@pytest.fixture(scope="function")
def avi_video(tmp_output_dir_func) -> Callable[[Tuple[int, int], int, bool], Path]:

    def _make_video(shape=(100, 50), frames=10, is_color=True) -> Path:
        shape = (frames, *shape)
        if is_color:
            shape = (*shape, 3)
        return VideoCase.make_array(
            shape=shape, dtype=np.uint8, path=tmp_output_dir_func
        )

    return _make_video
