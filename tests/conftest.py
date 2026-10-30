import sys

import pytest

from numpydantic.testing.cases import DTYPE_CASES, SHAPE_CASES
from numpydantic.testing.helpers import ValidationCase
from tests.fixtures import *

# python 3.12's generic syntax can't be parsed by older versions
collect_ignore = ["test_shape_typevars_pep695.py"] if sys.version_info < (3, 12) else []


def pytest_addoption(parser):
    parser.addoption(
        "--with-output",
        action="store_true",
        help="Keep test outputs in the __tmp__ directory",
    )


@pytest.fixture(
    scope="function",
    params=[pytest.param(c, id=c.id, marks=c.pytest_marks) for c in SHAPE_CASES],
)
def shape_cases(request, tmp_output_dir_func) -> ValidationCase:
    case: ValidationCase = request.param.model_copy()
    case.path = tmp_output_dir_func
    return case


@pytest.fixture(
    scope="function",
    params=[pytest.param(c, id=c.id, marks=c.pytest_marks) for c in DTYPE_CASES],
)
def dtype_cases(request, tmp_output_dir_func) -> ValidationCase:
    case: ValidationCase = request.param.model_copy()
    case.path = tmp_output_dir_func
    return case
