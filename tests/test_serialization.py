"""
Test serialization-specific functionality that doesn't need to be
applied across every interface (use test_interface/test_interfaces for that
"""

import json
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from numpydantic.serialization import _relativize_paths, _walk_and_apply, relative_path

pytestmark = pytest.mark.serialization


@pytest.fixture(scope="module")
def hdf5_at_path() -> Callable[[Path], None]:
    _path = ""

    def _hdf5_at_path(path: Path) -> None:
        nonlocal _path
        _path = path
        h5f = h5py.File(path, "w")
        _ = h5f.create_dataset("/data", data=np.array([[1, 2], [3, 4]]))
        _ = h5f.create_dataset("subpath/to/dataset", data=np.array([[1, 2], [4, 5]]))
        h5f.close()

    yield _hdf5_at_path

    Path(_path).unlink(missing_ok=True)


def test_relative_path(hdf5_at_path, tmp_output_dir, model_blank):
    """
    By default, we should make all paths relative to the cwd
    """
    out_path = tmp_output_dir / "relative.h5"
    hdf5_at_path(out_path)
    model = model_blank(array=(out_path, "/data"))
    rt = model.model_dump_json(round_trip=True)
    file = json.loads(rt)["array"]["file"]

    # should not be absolute
    assert not Path(file).is_absolute()
    # should be relative to cwd
    out_file = (Path.cwd() / file).resolve()
    assert out_file == out_path.resolve()


def test_relative_to_path(hdf5_at_path, tmp_output_dir, model_blank):
    """
    When explicitly passed a path to be ``relative_to`` ,
    relative to that instead of cwd
    """
    out_path = tmp_output_dir / "relative.h5"
    relative_to_path = Path(__file__) / "fake_dir" / "sub_fake_dir"
    expected_path = Path("../../../__tmp__/relative.h5")

    hdf5_at_path(out_path)
    model = model_blank(array=(out_path, "/data"))
    rt = model.model_dump_json(
        round_trip=True, context={"relative_to": str(relative_to_path)}
    )
    data = json.loads(rt)["array"]
    file = data["file"]

    # should not be absolute
    assert not Path(file).is_absolute()
    # should be expected path and reach the file
    assert Path(file) == expected_path
    assert (relative_to_path / file).resolve() == out_path.resolve()

    # we shouldn't have touched `/data` even though it is pathlike
    assert data["path"] == "/data"


def test_relative_to_root_dir():
    """
    The relativize function should ignore paths that are directories
    beneath the root directory (eg `/data`) even if they exist

    """
    # python 3.9 compat, which can't use negative indices
    test_path = [p for p in Path(__file__).resolve().parents][-2]

    test_data = {"some_field": str(test_path)}

    walked = _relativize_paths(test_data, relative_to=".")
    assert str(relative_path(test_path, Path(".").resolve())) != str(test_path)
    assert walked["some_field"] == str(test_path)


def test_absolute_path(hdf5_at_path, tmp_output_dir, model_blank):
    """
    When told, we make paths absolute
    """
    out_path = tmp_output_dir / "relative.h5"
    expected_dataset = "subpath/to/dataset"

    hdf5_at_path(out_path)
    model = model_blank(array=(out_path, expected_dataset))
    rt = model.model_dump_json(round_trip=True, context={"absolute_paths": True})
    data = json.loads(rt)["array"]
    file = data["file"]

    # should be absolute and equal to out_path
    assert Path(file).is_absolute()
    assert Path(file) == out_path.resolve()

    # shouldn't have absolutized subpath even if it's pathlike
    assert data["path"] == expected_dataset


def test_walk_and_apply():
    """
    Walk and apply should recursively apply a function to everything in a
    nesty structure
    """
    test = {
        "a": 1,
        "b": 1,
        "c": [
            {"a": 1, "b": {"a": 1, "b": 1}, "c": [1, 1, 1]},
            {"a": 1, "b": [1, 1, 1]},
        ],
    }

    def _mult_2(v, skip: bool = False):
        return v * 2

    def _assert_2(v, skip: bool = False):
        assert v == 2
        return v

    walked = _walk_and_apply(test, _mult_2)
    _walk_and_apply(walked, _assert_2)

    assert walked["a"] == 2
    assert walked["c"][0]["a"] == 2
    assert walked["c"][0]["b"]["a"] == 2
    assert all([w == 2 for w in walked["c"][0]["c"]])
    assert walked["c"][1]["a"] == 2
    assert all([w == 2 for w in walked["c"][1]["b"]])
