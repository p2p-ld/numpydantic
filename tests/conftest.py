import pytest

from numpydantic.testing.cases import DTYPE_CASES, SHAPE_CASES
from numpydantic.testing.helpers import ValidationCase
from tests.fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--with-output",
        action="store_true",
        help="Keep test outputs in the __tmp__ directory",
    )


@pytest.fixture(
    scope="function", params=[pytest.param(c, id=c.id) for c in SHAPE_CASES]
)
def shape_cases(request, tmp_output_dir_func) -> ValidationCase:
    case: ValidationCase = request.param.model_copy()
    case.path = tmp_output_dir_func
    return case


@pytest.fixture(
    scope="function", params=[pytest.param(c, id=c.id) for c in DTYPE_CASES]
)
def dtype_cases(request, tmp_output_dir_func) -> ValidationCase:
    case: ValidationCase = request.param.model_copy()
    case.path = tmp_output_dir_func
    return case
