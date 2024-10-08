import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type, Union
from warnings import warn
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, Field
import zarr
import cv2

from numpydantic.interface.hdf5 import H5ArrayPath
from numpydantic.interface.zarr import ZarrArrayPath
from numpydantic import NDArray, Shape
from numpydantic.maps import python_to_nptyping
from numpydantic.dtype import Number


@pytest.fixture(scope="session")
def tmp_output_dir(request: pytest.FixtureRequest) -> Path:
    path = Path(__file__).parent.resolve() / "__tmp__"
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir()

    yield path

    if not request.config.getvalue("--with-output"):
        try:
            shutil.rmtree(str(path))
        except PermissionError as e:
            # sporadic error on windows machines...
            warn(
                f"Temporary directory could not be removed due to a permissions error: \n{str(e)}"
            )


@pytest.fixture(scope="function")
def tmp_output_dir_func(tmp_output_dir, request: pytest.FixtureRequest) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / f"__tmpfunc_{request.node.name}__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath


@pytest.fixture(scope="module")
def tmp_output_dir_mod(tmp_output_dir, request: pytest.FixtureRequest) -> Path:
    """
    tmp output dir that gets cleared between every function
    cleans at the start rather than at cleanup in case the output is to be inspected
    """
    subpath = tmp_output_dir / f"__tmpmod_{request.module}__"
    if subpath.exists():
        shutil.rmtree(str(subpath))
    subpath.mkdir()
    return subpath


@pytest.fixture(scope="function")
def array_model() -> (
    Callable[[Tuple[int, ...], Union[Type, np.dtype]], Type[BaseModel]]
):
    def _model(
        shape: Tuple[int, ...] = (10, 10), dtype: Union[Type, np.dtype] = float
    ) -> Type[BaseModel]:
        shape_str = ", ".join([str(s) for s in shape])

        class MyModel(BaseModel):
            array: NDArray[Shape[shape_str], dtype]

        return MyModel

    return _model


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


@pytest.fixture(scope="session")
def model_blank() -> Type[BaseModel]:
    """A model with any shape and dtype"""

    class BlankModel(BaseModel):
        array: NDArray[Shape["*, ..."], Any]

    return BlankModel


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

        if not compound:
            if dtype is str:
                data = np.random.random(shape).astype(bytes)
            elif dtype is datetime:
                data = np.empty(shape, dtype="S32")
                data.fill(datetime.now(timezone.utc).isoformat().encode("utf-8"))
            else:
                data = np.random.random(shape).astype(dtype)

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
    array = root.zeros(path, shape=(100, 100), chunks=(10, 10))
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
