from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Tuple, Union

import cv2
import h5py
import numpy as np
import pytest
import zarr

from numpydantic.interface.hdf5 import H5ArrayPath
from numpydantic.interface.zarr import ZarrArrayPath


@pytest.fixture(scope="function")
def hdf5_array(
    request, tmp_output_dir_func
) -> Callable[[Tuple[int, ...], Union[np.dtype, type]], H5ArrayPath]:
    hdf5_file = tmp_output_dir_func / "h5f.h5"

    def _hdf5_array(
        shape: Tuple[int, ...] = (10, 10),
        dtype: Union[np.dtype, type] = float,
        compound: bool = False,
    ) -> H5ArrayPath:
        array_path = "/" + "_".join([str(s) for s in shape]) + "__" + dtype.__name__
        generator = np.random.default_rng()
        
        if not compound:
            if dtype is str:
                data = generator.random(shape).astype(bytes)
            elif dtype is datetime:
                data = np.empty(shape, dtype="S32")
                data.fill(datetime.now(timezone.utc).isoformat().encode("utf-8"))
            else:
                data = generator.random(shape).astype(dtype)

            h5path = H5ArrayPath(hdf5_file, array_path)
        else:
            if dtype is str:
                dt = np.dtype([("data", np.dtype("S10")), ("extra", "i8")])
                data = np.array([("hey", 0)] * np.prod(shape), dtype=dt).reshape(shape)
            elif dtype is datetime:
                dt = np.dtype([("data", np.dtype("S32")), ("extra", "i8")])
                data = np.array(
                    [(datetime.now(timezone.utc).isoformat().encode("utf-8"), 0)]
                    * np.prod(shape),
                    dtype=dt,
                ).reshape(shape)
            else:
                dt = np.dtype([("data", dtype), ("extra", "i8")])
                data = np.zeros(shape, dtype=dt)
            h5path = H5ArrayPath(hdf5_file, array_path, "data")

        with h5py.File(hdf5_file, "w") as h5f:
            _ = h5f.create_dataset(array_path, data=data)
        return h5path

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
    video_path = tmp_output_dir_func / "test.avi"

    def _make_video(shape=(100, 50), frames=10, is_color=True) -> Path:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"RGBA"),  # raw video for testing purposes
            30,
            (shape[1], shape[0]),
            is_color,
        )
        if is_color:
            shape = (*shape, 3)

        for i in range(frames):
            # make fresh array every time bc opencv eats them
            array = np.zeros(shape, dtype=np.uint8)
            if not is_color:
                array[i, i] = i
            else:
                array[i, i, :] = i
            writer.write(array)
        writer.release()
        return video_path

    return _make_video
