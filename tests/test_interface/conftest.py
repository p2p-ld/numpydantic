import pytest

from numpydantic.testing.cases import (
    ALL_CASES,
    ALL_CASES_PASSING,
    DTYPE_AND_INTERFACE_CASES_PASSING,
)
from numpydantic.testing.helpers import InterfaceCase, ValidationCase, merge_cases
from numpydantic.testing.interfaces import (
    DaskCase,
    HDF5Case,
    NumpyCase,
    VideoCase,
    ZarrCase,
    ZarrDirCase,
    ZarrNestedCase,
)


@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            NumpyCase,
            marks=pytest.mark.numpy,
            id="numpy",
        ),
        pytest.param(
            HDF5Case,
            marks=pytest.mark.hdf5,
            id="h5-array-path",
        ),
        pytest.param(
            DaskCase,
            marks=pytest.mark.dask,
            id="dask",
        ),
        pytest.param(
            ZarrCase,
            marks=pytest.mark.zarr,
            id="zarr-memory",
        ),
        pytest.param(
            ZarrNestedCase,
            marks=pytest.mark.zarr,
            id="zarr-nested",
        ),
        pytest.param(
            ZarrDirCase,
            marks=pytest.mark.zarr,
            id="zarr-dir",
        ),
        pytest.param(VideoCase, marks=pytest.mark.video, id="video"),
    ],
)
def interface_cases(request) -> InterfaceCase:
    """
    Fixture for combinatoric tests across all interface cases
    """
    return request.param


@pytest.fixture(
    params=(
        pytest.param(p, id=p.id, marks=getattr(pytest.mark, p.interface.interface.name))
        for p in ALL_CASES
    )
)
def all_cases(interface_cases, request) -> ValidationCase:
    """
    Combinatoric testing for all dtype, shape, and interface cases.

    This is a very expensive fixture! Only use it for core functionality
    that we want to be sure is *very true* in every circumstance,
    INCLUDING invalid combinations of annotations and arrays.
    Typically, that means only use this in `test_interfaces.py`
    """

    case = merge_cases(request.param, ValidationCase(interface=interface_cases))
    if case.skip():
        pytest.skip()
    return case


@pytest.fixture(
    params=(
        pytest.param(p, id=p.id, marks=getattr(pytest.mark, p.interface.interface.name))
        for p in ALL_CASES_PASSING
    )
)
def all_passing_cases(request) -> ValidationCase:
    """
    Combinatoric testing for all dtype, shape, and interface cases,
    but only the combinations that we expect to pass.

    This is a very expensive fixture! Only use it for core functionality
    that we want to be sure is *very true* in every circumstance.
    Typically, that means only use this in `test_interfaces.py`
    """
    return request.param


@pytest.fixture()
def all_cases_instance(all_cases, tmp_output_dir_func):
    """
    all_cases but with an instantiated model
    Args:
        all_cases:

    Returns:

    """
    array = all_cases.array(path=tmp_output_dir_func)
    instance = all_cases.model(array=array)
    return instance


@pytest.fixture()
def all_passing_cases_instance(all_passing_cases, tmp_output_dir_func):
    """
    all_cases but with an instantiated model
    Args:
        all_cases:

    Returns:

    """
    array = all_passing_cases.array(path=tmp_output_dir_func)
    instance = all_passing_cases.model(array=array)
    return instance


@pytest.fixture(
    params=(
        pytest.param(p, id=p.id, marks=getattr(pytest.mark, p.interface.interface.name))
        for p in DTYPE_AND_INTERFACE_CASES_PASSING
    )
)
def dtype_by_interface(request):
    """
    Tests for all dtypes by all interfaces
    """
    return request.param


@pytest.fixture()
def dtype_by_interface_instance(dtype_by_interface, tmp_output_dir_func):
    array = dtype_by_interface.array(path=tmp_output_dir_func)
    instance = dtype_by_interface.model(array=array)
    return instance
