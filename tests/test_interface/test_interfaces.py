"""
Tests that should be applied to all interfaces
"""

import pytest
from typing import Callable
import numpy as np
import dask.array as da
from zarr.core import Array as ZarrArray
from numpydantic.interface import Interface


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
    json = all_interfaces.model_dump_json(round_trip=round_trip)
    model = all_interfaces.model_validate_json(json)
    if round_trip:
        assert type(model.array) is type(all_interfaces.array)
        if isinstance(all_interfaces.array, (np.ndarray, ZarrArray)):
            assert np.array_equal(model.array, np.array(all_interfaces.array))
        elif isinstance(all_interfaces.array, da.Array):
            assert np.all(da.equal(model.array, all_interfaces.array))
        else:
            assert model.array == all_interfaces.array

        assert model.array.dtype == all_interfaces.array.dtype
    else:
        assert np.array_equal(model.array, np.array(all_interfaces.array))


def test_dunder_len(all_interfaces):
    """
    Each interface or proxy type should support __len__
    """
    assert len(all_interfaces.array) == all_interfaces.array.shape[0]
