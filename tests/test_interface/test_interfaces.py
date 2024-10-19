"""
Tests that should be applied to all interfaces
"""

import json
from importlib.metadata import version
from typing import Generic, TypeVar, Union

import dask.array as da
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict
from zarr.core import Array as ZarrArray

from numpydantic import NDArray
from numpydantic.interface import Interface, InterfaceMark, MarkedJson
from numpydantic.testing.cases import (
    ALL_CASES_PASSING,
    DTYPE_AND_INTERFACE_CASES_PASSING,
    INTERFACE_CASES,
)
from numpydantic.testing.helpers import ValidationCase


def _test_roundtrip(source: BaseModel, target: BaseModel):
    """Test model equality for roundtrip tests"""

    assert type(target.array) is type(source.array)
    if isinstance(source.array, (np.ndarray, ZarrArray)):
        assert np.array_equal(target.array, np.array(source.array))
    elif isinstance(source.array, da.Array):
        if target.array.dtype == object:
            # object equality doesn't really work well with dask
            # just check that the types match
            target_type = type(target.array.ravel()[0].compute())
            source_type = type(source.array.ravel()[0].compute())
            assert target_type is source_type
        else:
            assert np.all(da.equal(target.array, source.array))
    elif isinstance(source.array, BaseModel):
        return _test_roundtrip(source.array, target.array)
    else:
        assert target.array == source.array

    assert target.array.dtype == source.array.dtype


@pytest.mark.parametrize(
    "interface",
    [
        pytest.param(i, marks=getattr(pytest.mark, i.interface.interface.name))
        for i in INTERFACE_CASES
    ],
)
@pytest.mark.parametrize(
    "cases", [ALL_CASES_PASSING, DTYPE_AND_INTERFACE_CASES_PASSING]
)
def test_cases_include_all_interfaces(interface: ValidationCase, cases):
    """
    Test our test cases - we should hit all interfaces in the common "test all" fixtures
    """
    cases = list(cases)
    assert any(
        [case.interface is interface.interface for case in cases]
    ), f"Interface case unused in general test cases: {interface.interface}"


def test_dunder_len(interface_cases, tmp_output_dir_func):
    """
    Each interface or proxy type should support __len__
    """
    case = ValidationCase(interface=interface_cases)
    if interface_cases.interface.name == "video":
        case.shape = (10, 10, 2, 3)
        case.dtype = np.uint8
        case.annotation_dtype = np.uint8
        case.annotation_shape = (10, 10, "*", 3)
    array = case.array(path=tmp_output_dir_func)
    instance = case.model(array=array)
    assert len(instance.array) == case.shape[0]


def test_interface_revalidate(all_passing_cases_instance):
    """
    An interface should revalidate with the output of its initial validation

    See: https://github.com/p2p-ld/numpydantic/pull/14
    """

    _ = type(all_passing_cases_instance)(array=all_passing_cases_instance.array)


@pytest.mark.xfail
def test_interface_rematch(interface_cases, tmp_output_dir_func):
    """
    All interfaces should match the results of the object they return after validation
    """
    array = interface_cases.make_array(path=tmp_output_dir_func)

    assert (
        Interface.match(interface_cases.interface.validate(array))
        is interface_cases.interface
    )


def test_interface_to_numpy_array(dtype_by_interface_instance):
    """
    All interfaces should be able to have the output of their validation stage
    coerced to a numpy array with np.array()
    """
    _ = np.array(dtype_by_interface_instance.array)


@pytest.mark.serialization
def test_interface_dump_json(dtype_by_interface_instance):
    """
    All interfaces should be able to dump to json
    """
    dtype_by_interface_instance.model_dump_json()


@pytest.mark.serialization
def test_interface_roundtrip_json(all_passing_cases, tmp_output_dir_func):
    """
    All interfaces should be able to roundtrip to and from json
    """

    array = all_passing_cases.array(path=tmp_output_dir_func)
    case = all_passing_cases.model(array=array)

    dumped_json = case.model_dump_json(round_trip=True)
    model = case.model_validate_json(dumped_json)
    _test_roundtrip(case, model)


@pytest.mark.serialization
@pytest.mark.parametrize("an_interface", Interface.interfaces())
def test_interface_mark_interface(an_interface):
    """
    All interfaces should be able to mark the current version and interface info
    """
    mark = an_interface.mark_interface()
    assert isinstance(mark, InterfaceMark)
    assert mark.name == an_interface.name
    assert mark.cls == an_interface.__name__
    assert mark.module == an_interface.__module__
    assert mark.version == version(mark.module.split(".")[0])


@pytest.mark.serialization
@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.filterwarnings("ignore:Mismatch between serialized mark")
def test_interface_mark_roundtrip(all_passing_cases, valid, tmp_output_dir_func):
    """
    All interfaces should be able to roundtrip with the marked interface,
    and a mismatch should raise a warning and attempt to proceed
    """
    if "subclass" in all_passing_cases.id.lower():
        pytest.xfail()

    array = all_passing_cases.array(path=tmp_output_dir_func)
    case = all_passing_cases.model(array=array)

    dumped_json = case.model_dump_json(
        round_trip=True, context={"mark_interface": True}
    )

    data = json.loads(dumped_json)

    # ensure that we are a MarkedJson
    _ = MarkedJson.model_validate_json(json.dumps(data["array"]))

    if not valid:
        # ruin the version
        data["array"]["interface"]["version"] = "v99999999"
        dumped_json = json.dumps(data)

        with pytest.warns(match="Mismatch.*"):
            model = case.model_validate_json(dumped_json)
    else:
        model = case.model_validate_json(dumped_json)

    _test_roundtrip(case, model)


@pytest.mark.serialization
def test_roundtrip_from_extra(dtype_by_interface, tmp_output_dir_func):
    """
    Arrays can be dumped when they are specified in an `__extra__` field
    """

    class Model(BaseModel):
        __pydantic_extra__: dict[str, dtype_by_interface.annotation]
        model_config = ConfigDict(extra="allow")

    instance = Model(array=dtype_by_interface.array(path=tmp_output_dir_func))
    dumped = instance.model_dump_json(round_trip=True)
    roundtripped = Model.model_validate_json(dumped)
    _test_roundtrip(instance, roundtripped)


@pytest.mark.serialization
def test_roundtrip_from_union(dtype_by_interface, tmp_output_dir_func):
    """
    Arrays can be dumped when they are specified along with a
    union of another type field
    """

    class Model(BaseModel):
        array: Union[str, dtype_by_interface.annotation]

    array = dtype_by_interface.array(path=tmp_output_dir_func)

    instance = Model(array=array)
    dumped = instance.model_dump_json(round_trip=True)
    roundtripped = Model.model_validate_json(dumped)
    _test_roundtrip(instance, roundtripped)


@pytest.mark.serialization
def test_roundtrip_from_generic(dtype_by_interface, tmp_output_dir_func):
    """
    Arrays can be dumped when they are specified in an `__extra__` field
    """
    T = TypeVar("T", bound=NDArray)

    class GenType(BaseModel, Generic[T]):
        array: T

    class Model(BaseModel):
        array: GenType[dtype_by_interface.annotation]

    array = dtype_by_interface.array(path=tmp_output_dir_func)
    instance = Model(**{"array": {"array": array}})
    dumped = instance.model_dump_json(round_trip=True)
    roundtripped = Model.model_validate_json(dumped)
    _test_roundtrip(instance, roundtripped)
