import pdb
import json
import pytest

from pydantic import BaseModel, ValidationError

from numpydantic.interface import H5Interface
from numpydantic.interface.hdf5 import H5ArrayPath
from numpydantic.exceptions import DtypeError, ShapeError

from tests.conftest import ValidationCase


def hdf5_array_case(case: ValidationCase, array_func) -> H5ArrayPath:
    """
    Args:
        case:
        array_func: ( the function returned from the `hdf5_array` fixture )

    Returns:

    """
    return array_func(case.shape, case.dtype)


def _test_hdf5_case(case: ValidationCase, array_func):
    array = hdf5_array_case(case, array_func)
    if case.passes:
        case.model(array=array)
    else:
        with pytest.raises((ValidationError, DtypeError, ShapeError)):
            case.model(array=array)


def test_hdf5_enabled():
    assert H5Interface.enabled()


def test_hdf5_check(interface_type):
    if interface_type[1] is H5Interface:
        if interface_type[0].__name__ == "_hdf5_array":
            interface_type = (interface_type[0](), interface_type[1])
        assert H5Interface.check(interface_type[0])
        if isinstance(interface_type[0], H5ArrayPath):
            # also test that we can instantiate from a tuple like the H5ArrayPath
            assert H5Interface.check((interface_type[0].file, interface_type[0].path))
    else:
        assert not H5Interface.check(interface_type[0])


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


def test_hdf5_shape(shape_cases, hdf5_array):
    _test_hdf5_case(shape_cases, hdf5_array)


def test_hdf5_dtype(dtype_cases, hdf5_array):
    if dtype_cases.dtype is str:
        pytest.skip("hdf5 cant do string arrays")
    _test_hdf5_case(dtype_cases, hdf5_array)


def test_hdf5_dataset_not_exists(hdf5_array, model_blank):
    array = hdf5_array()
    with pytest.raises(ValueError) as e:
        model_blank(array=H5ArrayPath(file=array.file, path="/some/random/path"))
        assert "file located" in e
        assert "no array found" in e


def test_assignment(hdf5_array, model_blank):
    array = hdf5_array()

    model = model_blank(array=array)
    model.array[1, 1] = 5
    assert model.array[1, 1] == 5

    model.array[1:3, 2:4] = 10
    assert (model.array[1:3, 2:4] == 10).all()


def test_to_json(hdf5_array, array_model):
    """
    Test serialization of HDF5 arrays to JSON
    Args:
        hdf5_array:

    Returns:

    """
    array = hdf5_array((10, 10), int)
    model = array_model((10, 10), int)

    instance = model(array=array)  # type: BaseModel

    json_str = instance.model_dump_json()
    json_dict = json.loads(json_str)["array"]

    assert json_dict["file"] == str(array.file)
    assert json_dict["path"] == str(array.path)
    assert json_dict["attrs"] == {}
    assert json_dict["array"] == instance.array[:].tolist()
