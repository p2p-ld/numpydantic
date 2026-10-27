import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from pydantic import BaseModel
from zarr.storage import ZipStore

from numpydantic import AnyDims, NDArray, Shape
from numpydantic.interface import ZarrInterface
from numpydantic.interface import zarr as zarr_interface
from numpydantic.interface.zarr import ZarrArrayPath, ZarrJsonDict, hexdigest
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


@pytest.mark.parametrize("nested", [False, True])
def test_zarr_from_zip_path(nested, tmp_output_dir_func, model_blank):
    """Paths to .zip files should be opened as zip stores"""
    file = tmp_output_dir_func / "array.zip"
    path = "a/b/c" if nested else None
    store = ZipStore(file, mode="w")
    zarr.array(np.arange(10), store=store, path=path)
    store.close()

    assert ZarrInterface.check((file, path))
    instance = model_blank(array=(file, path))
    assert np.array_equal(instance.array[:], np.arange(10))


@pytest.mark.parametrize("dump_array", [True, False])
@pytest.mark.parametrize("dtype", [int, str])
def test_zarr_v2_format(dtype, dump_array, tmp_output_dir_func):
    """Arrays written in the zarr 2 format should still validate and roundtrip"""

    class Model(BaseModel):
        array: NDArray[Shape[*AnyDims], dtype]

    file = tmp_output_dir_func / "v2.zarr"
    data = np.array([[1, 2], [3, 4]]).astype(str if dtype is str else int)
    zarr.array(data, dtype=dtype, store=str(file), path="a/b", zarr_format=2)

    instance = Model(array=(file, "a/b"))
    assert instance.array.metadata.zarr_format == 2
    assert np.array_equal(instance.array[:], data)

    dumped = instance.model_dump_json(
        round_trip=True, context={"dump_array": dump_array}
    )
    reloaded = Model.model_validate_json(dumped)
    assert np.array_equal(reloaded.array[:], data)


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
@pytest.mark.parametrize("hash_array", [True, False])
@pytest.mark.parametrize("dump_array", [True, False])
@pytest.mark.parametrize("roundtrip", [True, False])
def test_zarr_to_json(
    zarr_case, model_blank, roundtrip, dump_array, hash_array, tmp_path
):
    lol_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)

    array = zarr_case.make_array(array=lol_array, dtype=int, path=tmp_path)
    instance = model_blank(array=array)
    expected_info = json.loads(json.dumps(instance.array.metadata.to_dict()))

    context = {"dump_array": dump_array, "hash_array": hash_array}
    as_json = json.loads(
        instance.model_dump_json(round_trip=roundtrip, context=context)
    )["array"]

    if roundtrip:
        if dump_array:
            assert np.array_equal(as_json["value"], lol_array)
        else:
            if as_json.get("file", False):
                assert "array" not in as_json

        assert as_json["info"] == expected_info
        assert as_json["info"]["shape"] == [3, 3]
        if hash_array:
            assert as_json["hexdigest"] == hexdigest(instance.array)
        else:
            assert "hexdigest" not in as_json

    else:
        assert np.array_equal(as_json, lol_array)


def _digest_from_files(directory: Path) -> str:
    """Compute the hexdigest from the files on disk, without zarr"""
    files = sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )
    digest = hashlib.sha256()
    for f in files:
        digest.update((directory / f).read_bytes())
    return digest.hexdigest()


@pytest.mark.serialization
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("nested", [False, True])
def test_zarr_hexdigest_from_files(nested, zarr_format, tmp_path):
    """
    The hexdigest should be reproducible from the files on disk, without zarr.

    A *little* circular as a test, mostly checking OS level ordering thingz,
    whether a naive `sorted(Path().rglob())` get us
    what we get from sorting zarr's manifest.
    """
    data = np.arange(100).reshape(10, 10)
    if nested:
        group = zarr.open_group(tmp_path / "g.zarr", mode="w", zarr_format=zarr_format)
        array = group.create_array("a/b", data=data, chunks=(3, 4))
        directory = tmp_path / "g.zarr" / "a" / "b"
    else:
        array = zarr.create_array(
            tmp_path / "a.zarr", data=data, chunks=(3, 4), zarr_format=zarr_format
        )
        directory = tmp_path / "a.zarr"

    assert hexdigest(array) == _digest_from_files(directory)


@pytest.mark.serialization
def test_zarr_hash_array_off_by_default(model_blank):
    """Hashing reads every file, so only do it when asked"""
    instance = model_blank(array=zarr.zeros((2, 2)))
    dumped = json.loads(instance.model_dump_json(round_trip=True))
    assert "hexdigest" not in dumped["array"]


@pytest.mark.serialization
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("fill_value", [0.0, np.nan, -np.inf])
def test_zarr_json_metadata_roundtrip(zarr_format, fill_value):
    """Array metadata should come back unchanged after a trip through JSON"""
    array = zarr.full((0, 3), fill_value, dtype="f8", zarr_format=zarr_format)
    dumped = ZarrJsonDict(type="zarr", info=array.metadata).model_dump_json()
    reloaded = ZarrJsonDict.model_validate_json(dumped)
    assert isinstance(reloaded.info, type(array.metadata))
    # compare dumps rather than metadata, since NaN fill values aren't equal to
    # themselves in older versions of zarr
    assert reloaded.model_dump_json() == dumped


@pytest.mark.serialization
@pytest.mark.parametrize(
    "info", [{"Type": "Array"}, "not metadata", {"zarr_format": 3}]
)
def test_zarr_json_invalid_metadata(info):
    """Things that aren't zarr metadata should fail validation, not raise errors"""
    assert not ZarrJsonDict.is_valid({"type": "zarr", "info": info})


def test_zarr_json_schema():
    """
    The JSON schema should describe metadata with zarr's TypedDict,
    when pydantic can read it with the installed versions
    """
    info = ZarrJsonDict.model_json_schema()["properties"]["info"]
    if zarr_interface._metadata_dict_schema() is None:
        assert info["type"] == "object"
    else:
        assert info["$ref"] == "#/$defs/ArrayMetadataDict"


def test_zarr_json_schema_fallback(monkeypatch):
    """If pydantic can't read zarr's TypedDict, metadata should be any object"""
    monkeypatch.setattr(zarr_interface, "_metadata_dict_schema", lambda: None)
    info = ZarrJsonDict.model_json_schema()["properties"]["info"]
    assert info["type"] == "object"
    assert "$defs" not in ZarrJsonDict.model_json_schema()
