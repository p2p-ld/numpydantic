"""
Tests that should be applied to all interfaces
"""

from typing import Callable
import numpy as np
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


def test_interface_dump_json(all_interfaces):
    """
    All interfaces should be able to dump to json
    """
    all_interfaces.model_dump_json()
