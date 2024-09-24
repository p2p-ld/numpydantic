"""
Tests that should be applied to all interfaces
"""

import pytest
from typing import Callable
from importlib.metadata import version
import json

import numpy as np
import dask.array as da
from zarr.core import Array as ZarrArray
from pydantic import BaseModel

from numpydantic.interface import Interface, InterfaceMark, MarkedJson


def _test_roundtrip(source: BaseModel, target: BaseModel, round_trip: bool):
    """Test model equality for roundtrip tests"""
    if round_trip:
        assert type(target.array) is type(source.array)
        if isinstance(source.array, (np.ndarray, ZarrArray)):
            assert np.array_equal(target.array, np.array(source.array))
        elif isinstance(source.array, da.Array):
            assert np.all(da.equal(target.array, source.array))
        else:
            assert target.array == source.array

        assert target.array.dtype == source.array.dtype
    else:
        assert np.array_equal(target.array, np.array(source.array))


def test_dunder_len(all_interfaces):
    """
    Each interface or proxy type should support __len__
    """
    assert len(all_interfaces.array) == all_interfaces.array.shape[0]


def test_interface_revalidate(all_interfaces):
    """
    An interface should revalidate with the output of its initial validation

    See: https://github.com/p2p-ld/numpydantic/pull/14
    """
    _ = type(all_interfaces)(array=all_interfaces.array)


def test_interface_rematch(interface_type):
    """
    All interfaces should match the results of the object they return after validation
    """
    array, interface = interface_type
    if isinstance(array, Callable):
        array = array()

    assert Interface.match(interface().validate(array)) is interface


def test_interface_to_numpy_array(all_interfaces):
    """
    All interfaces should be able to have the output of their validation stage
    coerced to a numpy array with np.array()
    """
    _ = np.array(all_interfaces.array)


@pytest.mark.serialization
def test_interface_dump_json(all_interfaces):
    """
    All interfaces should be able to dump to json
    """
    all_interfaces.model_dump_json()


@pytest.mark.serialization
@pytest.mark.parametrize("round_trip", [True, False])
def test_interface_roundtrip_json(all_interfaces, round_trip):
    """
    All interfaces should be able to roundtrip to and from json
    """
    dumped_json = all_interfaces.model_dump_json(round_trip=round_trip)
    model = all_interfaces.model_validate_json(dumped_json)
    _test_roundtrip(all_interfaces, model, round_trip)


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
@pytest.mark.parametrize("round_trip", [True, False])
@pytest.mark.filterwarnings("ignore:Mismatch between serialized mark")
def test_interface_mark_roundtrip(all_interfaces, valid, round_trip):
    """
    All interfaces should be able to roundtrip with the marked interface,
    and a mismatch should raise a warning and attempt to proceed
    """
    dumped_json = all_interfaces.model_dump_json(
        round_trip=round_trip, context={"mark_interface": True}
    )

    data = json.loads(dumped_json)

    # ensure that we are a MarkedJson
    _ = MarkedJson.model_validate_json(json.dumps(data["array"]))

    if not valid:
        # ruin the version
        data["array"]["interface"]["version"] = "v99999999"
        dumped_json = json.dumps(data)

        with pytest.warns(match="Mismatch.*"):
            model = all_interfaces.model_validate_json(dumped_json)
    else:
        model = all_interfaces.model_validate_json(dumped_json)

    _test_roundtrip(all_interfaces, model, round_trip)
