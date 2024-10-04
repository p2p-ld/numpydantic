"""
Tests for the interface base model,
for tests that should apply to all interfaces, use ``test_interfaces.py``
"""

import gc
from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError

from numpydantic.interface import (
    Interface,
    InterfaceMark,
    JsonDict,
    MarkedJson,
    NumpyInterface,
)
from numpydantic.interface.interface import V


class MyJsonDict(JsonDict):
    type: Literal["my_json_dict"]
    field: str
    number: int

    def to_array_input(self) -> V:
        dumped = self.model_dump()
        dumped["extra_input_param"] = True
        return dumped


@pytest.fixture(scope="module")
def interfaces():
    """Define test interfaces in this module, and delete afterwards"""
    interfaces_enabled = True

    class Interface1(Interface):
        input_types = (list,)
        return_type = tuple
        priority = 1000
        checked = False

        @classmethod
        def check(cls, array):
            cls.checked = True
            return isinstance(array, list)

        @classmethod
        def enabled(cls) -> bool:
            return interfaces_enabled

    Interface2 = type("Interface2", Interface1.__bases__, dict(Interface1.__dict__))
    Interface2.checked = False
    Interface2.priority = 999

    class Interface3(Interface1):
        priority = 998
        checked = False

        @classmethod
        def enabled(cls) -> bool:
            return False

    class Interface4(Interface3):
        priority = 997
        checked = False

        @classmethod
        def enabled(cls) -> bool:
            return interfaces_enabled

    class Interfaces:
        interface1 = Interface1
        interface2 = Interface2
        interface3 = Interface3
        interface4 = Interface4

    yield Interfaces
    # Interface.__subclasses__().remove(Interface1)
    # Interface.__subclasses__().remove(Interface2)
    del Interfaces
    del Interface1
    del Interface2
    del Interface3
    del Interface4
    interfaces_enabled = False
    gc.collect()


def test_interface_match_error(interfaces):
    """
    Test that `match` and `match_output` raises errors when no or multiple matches
    are found
    """
    with pytest.raises(ValueError) as e:
        Interface.match([1, 2, 3])
    assert "Interface1" in str(e.value)
    assert "Interface2" in str(e.value)

    with pytest.raises(ValueError) as e:
        Interface.match(([1, 2, 3], ["hey"]))
    assert "No matching interfaces" in str(e.value)

    with pytest.raises(ValueError) as e:
        Interface.match_output((1, 2, 3))
    assert "Interface1" in str(e.value)
    assert "Interface2" in str(e.value)

    with pytest.raises(ValueError) as e:
        Interface.match_output("hey")
    assert "No matching interfaces" in str(e.value)


def test_interface_match_fast(interfaces):
    """
    fast matching should return as soon as an interface is found
    and not raise an error for duplicates
    """
    Interface.interfaces()[0].checked = False
    Interface.interfaces()[1].checked = False
    # this doesnt' raise an error
    matched = Interface.match([1, 2, 3], fast=True)
    assert matched == Interface.interfaces()[0]
    assert Interface.interfaces()[0].checked
    assert not Interface.interfaces()[1].checked


def test_interface_enabled(interfaces):
    """
    An interface shouldn't be included if it's not enabled
    """
    assert not interfaces.interface3.enabled()
    assert interfaces.interface3 not in Interface.interfaces()


def test_interface_type_lists():
    """
    Seems like a silly test, but ensure that our return types and input types
    lists have all the class attrs
    """
    for interface in Interface.interfaces():

        if isinstance(interface.input_types, (list, tuple)):
            for atype in interface.input_types:
                assert atype in Interface.input_types()
        else:
            assert interface.input_types in Interface.input_types()

        if isinstance(interface.return_type, (list, tuple)):
            for atype in interface.return_type:
                assert atype in Interface.return_types()
        else:
            assert interface.return_type in Interface.return_types()


def test_interfaces_sorting():
    """
    Interfaces should be returned in descending order of priority
    """
    ifaces = Interface.interfaces()
    priorities = [i.priority for i in ifaces]
    assert (np.diff(priorities) <= 0).all()


def test_interface_with_disabled(interfaces):
    """
    Get all interfaces, even if not enabled
    """
    ifaces = Interface.interfaces(with_disabled=True)
    assert interfaces.interface3 in ifaces


def test_interface_recursive(interfaces):
    """
    Get all interfaces, including subclasses of subclasses
    """
    ifaces = Interface.interfaces()
    assert issubclass(interfaces.interface4, interfaces.interface3)
    assert issubclass(interfaces.interface3, interfaces.interface1)
    assert issubclass(interfaces.interface1, Interface)
    assert interfaces.interface4 in ifaces


@pytest.mark.serialization
def test_jsondict_is_valid():
    """
    A JsonDict should return a bool true/false if it is valid or not,
    and raise an error when requested
    """
    invalid = {"doesnt": "have", "the": "props"}
    valid = {"type": "my_json_dict", "field": "a_field", "number": 1}
    assert MyJsonDict.is_valid(valid)
    assert not MyJsonDict.is_valid(invalid)
    with pytest.raises(ValidationError):
        assert not MyJsonDict.is_valid(invalid, raise_on_error=True)


@pytest.mark.serialization
def test_jsondict_handle_input():
    """
    JsonDict should be able to parse a valid dict and return it to the input format
    """
    valid = {"type": "my_json_dict", "field": "a_field", "number": 1}
    instantiated = MyJsonDict(**valid)
    expected = {
        "type": "my_json_dict",
        "field": "a_field",
        "number": 1,
        "extra_input_param": True,
    }

    for item in (valid, instantiated):
        result = MyJsonDict.handle_input(item)
        assert result == expected


@pytest.mark.serialization
@pytest.mark.parametrize("interface", Interface.interfaces())
def test_interface_mark_match_by_name(interface):
    """
    Interface mark should match an interface by its name
    """
    # other parts don't matter
    mark = InterfaceMark(module="fake", cls="fake", version="fake", name=interface.name)
    fake_mark = InterfaceMark(
        module="fake", cls="fake", version="fake", name="also_fake"
    )
    assert mark.match_by_name() is interface
    assert fake_mark.match_by_name() is None


@pytest.mark.serialization
def test_marked_json_try_cast():
    """
    MarkedJson.try_cast should try and cast to a markedjson!
    returning the value unchanged if it's not a match
    """
    valid = {"interface": NumpyInterface.mark_interface(), "value": [[1, 2], [3, 4]]}
    invalid = [1, 2, 3, 4, 5]
    mimic = {"interface": "not really", "value": "still not really"}

    assert isinstance(MarkedJson.try_cast(valid), MarkedJson)
    assert MarkedJson.try_cast(invalid) is invalid
    assert MarkedJson.try_cast(mimic) is mimic
