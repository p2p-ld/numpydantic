import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Type, Union

import h5py
import numpy as np
import pytest
from pydantic import BaseModel, Field
import zarr

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
        shutil.rmtree(str(path))


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
def hdf5_file(tmp_output_dir_func) -> h5py.File:
    h5f_file = tmp_output_dir_func / "h5f.h5"
    h5f = h5py.File(h5f_file, "w")
    yield h5f
    h5f.close()


@pytest.fixture(scope="function")
def hdf5_array(
    hdf5_file, request
) -> Callable[[Tuple[int, ...], Union[np.dtype, type]], H5ArrayPath]:

    def _hdf5_array(
        shape: Tuple[int, ...] = (10, 10), dtype: Union[np.dtype, type] = float
    ) -> H5ArrayPath:
        array_path = "/" + "_".join([str(s) for s in shape]) + "__" + dtype.__name__
        data = np.random.random(shape).astype(dtype)
        _ = hdf5_file.create_dataset(array_path, data=data)
        return H5ArrayPath(Path(hdf5_file.filename), array_path)

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
