import pytest

from numpydantic.linkml import ArrayFormat, NWBLinkMLArraylike

from ..fixtures import nwb_linkml_array


def test_nwb_linkml_array(nwb_linkml_array):
    classdef, generated = nwb_linkml_array

    assert ArrayFormat.is_array(classdef)
    assert NWBLinkMLArraylike.check(classdef)
    assert ArrayFormat.get(classdef) is NWBLinkMLArraylike
    assert generated == NWBLinkMLArraylike.make(classdef)
