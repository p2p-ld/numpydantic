import pytest
import zarr

from pydantic import ValidationError

from numpydantic.interface import ZarrInterface


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


STORES = (
    dir_array,
    zip_array,
)
"""stores for single arrays"""


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


@pytest.mark.parametrize(
    "array,passes",
    [
        (zarr.zeros((5, 10)), True),
        (zarr.zeros((5, 10, 3)), True),
        (zarr.zeros((5, 10, 3, 4)), True),
        (zarr.zeros((5, 10, 4)), False),
        (zarr.zeros((5, 10, 3, 6)), False),
        (zarr.zeros((5, 10, 4, 6)), False),
    ],
)
def test_zarr_shape(model_rgb, array, passes):
    if passes:
        model_rgb(array=array)
    else:
        with pytest.raises(ValidationError):
            model_rgb(array=array)
