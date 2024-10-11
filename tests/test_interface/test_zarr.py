import json

import numpy as np
import pytest

from numpydantic.interface import ZarrInterface
from numpydantic.interface.zarr import ZarrArrayPath
from numpydantic.testing.cases import ZarrCase, ZarrDirCase, ZarrNestedCase, ZarrZipCase
from numpydantic.testing.helpers import InterfaceCase

pytestmark = pytest.mark.zarr


@pytest.fixture(
    params=[ZarrCase, ZarrZipCase, ZarrDirCase, ZarrNestedCase],
)
def zarr_case(request) -> InterfaceCase:
    return request.param


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


def test_zarr_check(interface_cases, tmp_output_dir_func):
    """
    We should only use the zarr interface for zarr-like things
    """
    array = interface_cases.make_array(path=tmp_output_dir_func)
    if interface_cases.interface is ZarrInterface:
        assert ZarrInterface.check(array)
    else:
        assert not ZarrInterface.check(array)


@pytest.mark.shape
def test_zarr_shape(shape_cases, zarr_case):
    shape_cases.interface = zarr_case
    shape_cases.validate_case()


@pytest.mark.dtype
def test_zarr_dtype(dtype_cases, zarr_case):
    dtype_cases.interface = zarr_case
    if dtype_cases.skip():
        pytest.skip()
    dtype_cases.validate_case()


@pytest.mark.parametrize("array", ["zarr_nested_array", "zarr_array"])
def test_zarr_from_tuple(array, model_blank, request):
    """Should be able to do the same validation logic from tuples as an input"""
    array = request.getfixturevalue(array)
    if isinstance(array, ZarrArrayPath):
        _ = model_blank(array=(array.file, array.path))
    else:
        _ = model_blank(array=(array,))


def test_zarr_from_path(zarr_array, model_blank):
    """Should be able to just pass a path"""
    _ = model_blank(array=zarr_array)


def test_zarr_array_path_from_iterable(zarr_array):
    """Construct a zarr array path from some iterable!!!"""
    # from a single path
    apath = ZarrArrayPath.from_iterable((zarr_array,))
    assert apath.file == zarr_array
    assert apath.path is None

    inner_path = "/test/array"
    apath = ZarrArrayPath.from_iterable((zarr_array, inner_path))
    assert apath.file == zarr_array
    assert apath.path == inner_path


@pytest.mark.serialization
@pytest.mark.parametrize("dump_array", [True, False])
@pytest.mark.parametrize("roundtrip", [True, False])
def test_zarr_to_json(zarr_case, model_blank, roundtrip, dump_array, tmp_path):
    expected_fields = (
        "Type",
        "Data type",
        "Shape",
        "Chunk shape",
        "Compressor",
        "Store type",
        "hexdigest",
    )
    lol_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)

    array = zarr_case.make_array(array=lol_array, dtype=int, path=tmp_path)
    instance = model_blank(array=array)

    context = {"dump_array": dump_array}
    as_json = json.loads(
        instance.model_dump_json(round_trip=roundtrip, context=context)
    )["array"]

    if roundtrip:
        if dump_array:
            assert np.array_equal(as_json["value"], lol_array)
        else:
            if as_json.get("file", False):
                assert "array" not in as_json

        for field in expected_fields:
            assert field in as_json["info"]
        assert len(as_json["info"]["hexdigest"]) == 40

    else:
        assert np.array_equal(as_json, lol_array)
