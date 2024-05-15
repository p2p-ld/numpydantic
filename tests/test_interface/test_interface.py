import pytest

from numpydantic.interface import Interface


@pytest.fixture(scope="module")
def interfaces():
    """Define test interfaces in this module, and delete afterwards"""

    class Interface1(Interface):
        input_types = (list,)
        return_type = tuple

        @classmethod
        def check(cls, array):
            if isinstance(array, list):
                return True
            return False

        @classmethod
        def enabled(cls) -> bool:
            return True

    Interface2 = type("Interface2", Interface1.__bases__, dict(Interface1.__dict__))

    class Interface3(Interface1):
        @classmethod
        def enabled(cls) -> bool:
            return False

    class Interfaces:
        interface1 = Interface1
        interface2 = Interface2
        interface3 = Interface3

    yield Interfaces
    del Interface1
    del Interface2
    del Interface3


def test_interface_match_error(interfaces):
    """
    Test that `match` and `match_output` raises errors when no or multiple matches are found
    """
    with pytest.raises(ValueError) as e:
        Interface.match([1, 2, 3])
        assert "Interface1" in e
        assert "Interface2" in e

    with pytest.raises(ValueError) as e:
        Interface.match("hey")
        assert "No matching interfaces" in e

    with pytest.raises(ValueError) as e:
        Interface.match_output((1, 2, 3))
        assert "Interface1" in e
        assert "Interface2" in e

    with pytest.raises(ValueError) as e:
        Interface.match_output("hey")
        assert "No matching interfaces" in e


def test_interface_enabled(interfaces):
    """
    An interface shouldn't be included if it's not enabled
    """
    assert not interfaces.Interface3.enabled()
    assert interfaces.Interface3 not in Interface.interfaces()


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
