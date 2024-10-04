import sys

from numpydantic.testing.cases import YES_PIPE, RGB_UNION, UNION_PIPE, DTYPE_CASES, DTYPE_IDS

from numpydantic.testing.helpers import ValidationCase
from tests.fixtures import *


def pytest_addoption(parser):
    parser.addoption(
        "--with-output",
        action="store_true",
        help="Keep test outputs in the __tmp__ directory",
    )





@pytest.fixture(
    scope="module",
    params=[
        ValidationCase(shape=(10, 10, 10), passes=True),
        ValidationCase(shape=(10, 10), passes=False),
        ValidationCase(shape=(10, 10, 10, 10), passes=False),
        ValidationCase(shape=(11, 10, 10), passes=False),
        ValidationCase(shape=(9, 10, 10), passes=False),
        ValidationCase(shape=(10, 10, 9), passes=True),
        ValidationCase(shape=(10, 10, 11), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3, 4), passes=True),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 4), passes=False),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 3, 6), passes=False),
        ValidationCase(annotation=RGB_UNION, shape=(5, 5, 4, 6), passes=False),
    ],
    ids=[
        "valid shape",
        "missing dimension",
        "extra dimension",
        "dimension too large",
        "dimension too small",
        "wildcard smaller",
        "wildcard larger",
        "Union 2D",
        "Union 3D",
        "Union 4D",
        "Union incorrect 3D",
        "Union incorrect 4D",
        "Union incorrect both",
    ],
)
def shape_cases(request) -> ValidationCase:
    return request.param


@pytest.fixture(scope="module", params=DTYPE_CASES, ids=DTYPE_IDS)
def dtype_cases(request) -> ValidationCase:
    return request.param
