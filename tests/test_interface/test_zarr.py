import pytest
import zarr

from pydantic import ValidationError

from numpydantic.interface import ZarrInterface
from numpydantic.exceptions import DtypeError, ShapeError

from tests.conftest import ValidationCase


@pytest.fixture()
def dir_array(tmp_output_dir_func) -> zarr.DirectoryStore:
    store = zarr.DirectoryStore(tmp_output_dir_func / "array.zarr")
    return store


@pytest.fixture()
def zip_array(tmp_output_dir_func) -> zarr.ZipStore:
    store = zarr.ZipStore(tmp_output_dir_func / "array.zip", mode="w")
    return store


@pytest.fixture()
def nested_dir_array(tmp_output_dir_func) -> zarr.NestedDirectoryStore:
    store = zarr.NestedDirectoryStore(tmp_output_dir_func / "nested")
    return store


def zarr_array(case: ValidationCase, store) -> zarr.core.Array:
    return zarr.zeros(shape=case.shape, dtype=case.dtype, store=store)


def _test_zarr_case(case: ValidationCase, store):
    array = zarr_array(case, store)
    if case.passes:
        case.model(array=array)
    else:
        with pytest.raises((ValidationError, DtypeError, ShapeError)):
            case.model(array=array)


@pytest.fixture(
    params=[
        None,  # use the default store
        "dir_array",
        "zip_array",
        "nested_dir_array",
    ],
    ids=["MutableMapping", "DirectoryStore", "ZipStore", "NestedDirectoryStore"],
)
def store(request):
    if isinstance(request.param, str):
        return request.getfixturevalue(request.param)
    else:
        return request.param


def test_zarr_enabled():
    assert ZarrInterface.enabled()


def test_zarr_check(interface_type):
    """
    We should only use the zarr interface for zarr-like things
    """
    if interface_type[1] is ZarrInterface:
        assert ZarrInterface.check(interface_type[0])
    else:
        assert not ZarrInterface.check(interface_type[0])


def test_zarr_shape(store, shape_cases):
    _test_zarr_case(shape_cases, store)


def test_zarr_dtype(dtype_cases, store):
    _test_zarr_case(dtype_cases, store)
