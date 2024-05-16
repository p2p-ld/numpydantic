import sys
import pytest

from numpydantic import NDArray
from numpydantic.meta import update_ndarray_stub

if sys.version_info.minor < 11:
    from typing_extensions import reveal_type
else:
    from typing import reveal_type


def test_no_warn(recwarn):
    """
    If something is going wrong with generating meta stubs, a warning will be emitted.
    that is bad.
    """
    update_ndarray_stub()
    assert len(recwarn) == 0


@pytest.mark.skip("TODO")
def test_generate_stub():
    """
    Test that we generate the stub file correctly...
    """
    pass


@pytest.mark.skip("TODO")
def test_update_stub():
    """
    Test that the update stub file correctly updates the stub stored in the package
    """
    pass


@pytest.mark.skip("TODO")
def test_stub_revealed_type():
    """
    Check that the revealed type matches the stub
    """
    type = reveal_type(NDArray)
