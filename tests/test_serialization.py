"""
Test serialization-specific functionality that doesn't need to be
applied across every interface (use test_interface/test_interfaces for that
"""

import h5py
import pytest
from pathlib import Path
from typing import Callable
import numpy as np
import json

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
    expected_path = "../../../__tmp__/relative.h5"

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
    assert file == expected_path
    assert (relative_to_path / file).resolve() == out_path.resolve()

    # we shouldn't have touched `/data` even though it is pathlike
    assert data["path"] == "/data"


def test_relative_to_path(hdf5_at_path, tmp_output_dir, model_blank):
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
