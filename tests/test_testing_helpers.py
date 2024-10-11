"""
Tests for the testing helpers lmao
"""

import numpy as np
import pytest
from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.testing.cases import INTERFACE_CASES
from numpydantic.testing.helpers import ValidationCase
from numpydantic.testing.interfaces import NumpyCase


def test_validation_case_merge():
    case_1 = ValidationCase(id="1", interface=NumpyCase, passes=False)
    case_2 = ValidationCase(id="2", dtype=str, passes=True)
    case_3 = ValidationCase(id="3", shape=(1, 2, 3), passes=True)

    merged_simple = case_2.merge(case_3)
    assert merged_simple.dtype == case_2.dtype
    assert merged_simple.shape == case_3.shape

    merged_multi = case_1.merge([case_2, case_3])
    assert merged_multi.dtype == case_2.dtype
    assert merged_multi.shape == case_3.shape
    assert merged_multi.interface == case_1.interface

    # passes should be true only if all the cases are
    assert merged_simple.passes
    assert not merged_multi.passes

    # ids should merge
    assert merged_simple.id == "2-3"
    assert merged_multi.id == "1-2-3"


@pytest.mark.parametrize(
    "interface",
    [
        pytest.param(
            i.interface, marks=getattr(pytest.mark, i.interface.interface.name)
        )
        for i in INTERFACE_CASES
        if i.id not in ("hdf5_compound")
    ],
)
def test_make_array(interface, tmp_output_dir_func):
    """
    An interface case can generate an array from params or a given array

    Not testing correctness here, that's what hte rest of the testing does.
    """
    arr = np.zeros((10, 10, 2, 3), dtype=np.uint8)
    arr = interface.make_array(array=arr, dtype=np.uint8, path=tmp_output_dir_func)

    class MyModel(BaseModel):
        array: NDArray[Shape["10, 10, 2, 3"], np.uint8]

    _ = MyModel(array=arr)
