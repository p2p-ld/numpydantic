import json

import pytest
import zarr

from pydantic import BaseModel, ValidationError
from numcodecs import Pickle

from numpydantic.interface import ZarrInterface
from numpydantic.interface.zarr import ZarrArrayPath
from numpydantic.exceptions import DtypeError, ShapeError

from tests.conftest import ValidationCase

pytestmark = pytest.mark.zarr


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


def _zarr_array(case: ValidationCase, store) -> zarr.core.Array:
    if issubclass(case.dtype, BaseModel):
        pytest.skip(
            f"Zarr can't handle objects properly at the moment, "
            "see https://github.com/zarr-developers/zarr-python/issues/2081"
        )
        # return zarr.full(
        #     shape=case.shape,
        #     fill_value=case.dtype(x=1),
        #     dtype=object,
        #     object_codec=Pickle(),
        # )
    else:
        return zarr.zeros(shape=case.shape, dtype=case.dtype, store=store)


def _test_zarr_case(case: ValidationCase, store):
    array = _zarr_array(case, store)
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


@pytest.mark.shape
def test_zarr_shape(store, shape_cases):
    _test_zarr_case(shape_cases, store)


@pytest.mark.dtype
def test_zarr_dtype(dtype_cases, store):
    _test_zarr_case(dtype_cases, store)


@pytest.mark.parametrize("array", ["zarr_nested_array", "zarr_array"])
def test_zarr_from_tuple(array, model_blank, request):
    """Should be able to do the same validation logic from tuples as an input"""
    array = request.getfixturevalue(array)
    if isinstance(array, ZarrArrayPath):
        instance = model_blank(array=(array.file, array.path))
    else:
        instance = model_blank(array=(array,))


def test_zarr_from_path(zarr_array, model_blank):
    """Should be able to just pass a path"""
    instance = model_blank(array=zarr_array)


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
def test_zarr_to_json(store, model_blank, roundtrip, dump_array):
    expected_fields = (
        "Type",
        "Data type",
        "Shape",
        "Chunk shape",
        "Compressor",
        "Store type",
        "hexdigest",
    )
    lol_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    array = zarr.array(lol_array, store=store)
    instance = model_blank(array=array)

    context = {"dump_array": dump_array}
    as_json = json.loads(
        instance.model_dump_json(round_trip=roundtrip, context=context)
    )["array"]

    if roundtrip:
        if dump_array:
            assert as_json["value"] == lol_array
        else:
            if as_json.get("file", False):
                assert "array" not in as_json

        for field in expected_fields:
            assert field in as_json["info"]
        assert len(as_json["info"]["hexdigest"]) == 40

    else:
        assert as_json == lol_array
