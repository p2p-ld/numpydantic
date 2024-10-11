import json
from datetime import datetime
from typing import Any

import h5py
import numpy as np
import pytest
from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.interface import H5Interface
from numpydantic.interface.hdf5 import H5ArrayPath, H5Proxy
from numpydantic.testing.interfaces import HDF5Case, HDF5CompoundCase

pytestmark = pytest.mark.hdf5


@pytest.fixture(
    params=[
        pytest.param(HDF5Case, id="hdf5"),
        pytest.param(HDF5CompoundCase, id="hdf5-compound"),
    ]
)
def hdf5_cases(request):
    return request.param


def test_hdf5_enabled():
    assert H5Interface.enabled()


@pytest.mark.shape
def test_hdf5_shape(shape_cases, hdf5_cases):
    shape_cases.interface = hdf5_cases
    if shape_cases.skip():
        pytest.skip()
    shape_cases.validate_case()


@pytest.mark.dtype
def test_hdf5_dtype(dtype_cases, hdf5_cases):
    dtype_cases.interface = hdf5_cases
    dtype_cases.validate_case()


def test_hdf5_check(interface_cases, tmp_output_dir_func):
    array = interface_cases.make_array(path=tmp_output_dir_func)
    if interface_cases.interface is H5Interface:
        assert H5Interface.check(array)
    else:
        assert not H5Interface.check(array)


def test_hdf5_check_not_exists():
    """We should fail a check for a nonexistent hdf5 file"""
    spec = ("./fakefile.h5", "/fake/array")
    assert not H5Interface.check(spec)


def test_hdf5_check_not_hdf5(tmp_path):
    """Files that exist but aren't actually hdf5 files should fail a check"""
    afile = tmp_path / "not_an_hdf.h5"
    with open(afile, "w") as af:
        af.write("hey")

    spec = (afile, "/fake/array")
    assert not H5Interface.check(spec)


def test_hdf5_dataset_not_exists(hdf5_array, model_blank):
    array = hdf5_array()
    with pytest.raises(ValueError) as e:
        model_blank(array=H5ArrayPath(file=array.file, path="/some/random/path"))
        assert "file located" in e
        assert "no array found" in e


@pytest.mark.proxy
def test_assignment(hdf5_array, model_blank):
    array = hdf5_array()

    model = model_blank(array=array)
    model.array[1, 1] = 5
    assert model.array[1, 1] == 5

    model.array[1:3, 2:4] = 10
    assert (model.array[1:3, 2:4] == 10).all()


@pytest.mark.serialization
@pytest.mark.parametrize("round_trip", (True, False))
def test_to_json(hdf5_array, array_model, round_trip):
    """
    Test serialization of HDF5 arrays to JSON
    Args:
        hdf5_array:

    Returns:

    """
    array = hdf5_array((10, 10), int)
    model = array_model((10, 10), int)

    instance = model(array=array)  # type: BaseModel

    json_str = instance.model_dump_json(
        round_trip=round_trip, context={"absolute_paths": True}
    )
    json_dumped = json.loads(json_str)["array"]
    if round_trip:
        assert json_dumped["file"] == str(array.file)
        assert json_dumped["path"] == str(array.path)
    else:
        assert json_dumped == instance.array[:].tolist()


@pytest.mark.dtype
@pytest.mark.proxy
def test_compound_dtype(tmp_path):
    """
    hdf5 proxy indexes compound dtypes as single fields when field is given
    """
    h5f_path = tmp_path / "test.h5"
    dataset_path = "/dataset"
    field = "data"
    dtype = np.dtype([(field, "i8"), ("extra", "f8")])
    data = np.zeros((10, 20), dtype=dtype)
    with h5py.File(h5f_path, "w") as h5f:
        dset = h5f.create_dataset(dataset_path, data=data)
        assert dset.dtype == dtype

    proxy = H5Proxy(h5f_path, dataset_path, field=field)
    assert proxy.dtype == np.dtype("int64")
    assert proxy.shape == (10, 20)
    assert proxy[0, 0] == 0

    class MyModel(BaseModel):
        array: NDArray[Shape["10, 20"], np.int64]

    instance = MyModel(array=(h5f_path, dataset_path, field))
    assert instance.array.dtype == np.dtype("int64")
    assert instance.array.shape == (10, 20)
    assert instance.array[0, 0] == 0

    # set values too
    instance.array[0, :] = 1
    assert all(instance.array[0, :] == 1)
    assert all(instance.array[1, :] == 0)
    instance.array[1] = 2
    assert all(instance.array[1] == 2)


@pytest.mark.dtype
@pytest.mark.proxy
@pytest.mark.parametrize("compound", [True, False])
def test_strings(hdf5_array, compound):
    """
    HDF5 proxy can get and set strings just like any other dtype
    """
    array = hdf5_array((10, 10), str, compound=compound)

    class MyModel(BaseModel):
        array: NDArray[Shape["10, 10"], str]

    instance = MyModel(array=array)
    instance.array[0, 0] = "hey"
    assert instance.array[0, 0] == "hey"
    assert isinstance(instance.array[0, 1], str)

    instance.array[1] = "sup"
    assert all(instance.array[1] == "sup")


@pytest.mark.dtype
@pytest.mark.proxy
@pytest.mark.parametrize("compound", [True, False])
def test_datetime(hdf5_array, compound):
    """
    We can treat S32 byte arrays as datetimes if our type annotation
    says to, including validation, setting and getting values
    """
    array = hdf5_array((10, 10), datetime, compound=compound)

    class MyModel(BaseModel):
        array: NDArray[Any, datetime]

    instance = MyModel(array=array)
    assert isinstance(instance.array[0, 0], np.datetime64)
    assert instance.array[0:5].dtype.type is np.datetime64

    now = datetime.now()

    instance.array[0, 0] = now
    assert instance.array[0, 0] == now
    instance.array[0] = now
    assert all(instance.array[0] == now)


@pytest.mark.parametrize("dtype", [int, float, str, datetime])
def test_empty_dataset(dtype, tmp_path):
    """
    Empty datasets shouldn't choke us during validation
    """
    array_path = tmp_path / "test.h5"
    np_dtype = "S32" if dtype in (str, datetime) else dtype

    with h5py.File(array_path, "w") as h5f:
        _ = h5f.create_dataset(name="/data", dtype=np_dtype)

    class MyModel(BaseModel):
        array: NDArray[Any, dtype]

    _ = MyModel(array=(array_path, "/data"))


@pytest.mark.proxy
@pytest.mark.parametrize(
    "comparison,valid",
    [
        (H5Proxy(file="test_file.h5", path="/subpath", field="sup"), True),
        (H5Proxy(file="test_file.h5", path="/subpath"), False),
        (H5Proxy(file="different_file.h5", path="/subpath"), False),
        (("different_file.h5", "/subpath", "sup"), ValueError),
        ("not even a proxy-like thing", ValueError),
    ],
)
def test_proxy_eq(comparison, valid):
    """
    test the __eq__ method of H5ArrayProxy matches proxies to the same
    dataset (and path), or raises a ValueError
    """
    proxy_a = H5Proxy(file="test_file.h5", path="/subpath", field="sup")
    if valid is True:
        assert proxy_a == comparison
    elif valid is False:
        assert proxy_a != comparison
    else:
        with pytest.raises(valid):
            assert proxy_a == comparison
