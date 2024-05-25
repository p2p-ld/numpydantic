import pytest

import numpy as np

from numpydantic.interface import Interface


@pytest.fixture(scope="module")
def interfaces():
    """Define test interfaces in this module, and delete afterwards"""

    class Interface1(Interface):
        input_types = (list,)
        return_type = tuple
        priority = 1000
        checked = False

        @classmethod
        def check(cls, array):
            cls.checked = True
            if isinstance(array, list):
                return True
            return False

        @classmethod
        def enabled(cls) -> bool:
            return True

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
            return True

    class Interfaces:
        interface1 = Interface1
        interface2 = Interface2
        interface3 = Interface3
        interface4 = Interface4

    yield Interfaces
    # Interface.__subclasses__().remove(Interface1)
    # Interface.__subclasses__().remove(Interface2)
    del Interface1
    del Interface2
    del Interface3


def test_interface_match_error(interfaces):
    """
    Test that `match` and `match_output` raises errors when no or multiple matches are found
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
