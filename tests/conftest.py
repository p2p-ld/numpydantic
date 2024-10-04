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


@pytest.fixture(scope="module", params=SHAPE_CASES)
def shape_cases(request) -> ValidationCase:
    return request.param


@pytest.fixture(scope="module", params=DTYPE_CASES)
def dtype_cases(request) -> ValidationCase:
    return request.param
