"""
Shape typevars declared with python 3.12's syntax are validated the same way
as ones made with ``TypeVar``.

Only collected on python 3.12 and later, since the syntax can't be parsed before that.
"""

from typing import Any, Generic, TypeVar
from typing import Literal as L

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from numpydantic import NDArray, Range, Shape

pytestmark = pytest.mark.shape

F = np.float64
N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


class _Declared[
    N: int,
    RGB: (L[3], L[4]),
    OneToFive: (L[1], L[2, 3, 4, 5]),
    Small: Range[L[1], L[4]],
]:
    """Holds typevars declared with the 3.12 syntax"""


FORMS = (
    "int bound",
    "literal constraints",
    "union of literal constraints",
    "range bound",
)
TYPEVARS = {
    "TypeVar": dict(
        zip(
            FORMS,
            (
                TypeVar("N", bound=int),
                TypeVar("RGB", L[3], L[4]),
                TypeVar("OneToFive", L[1], L[2, 3, 4, 5]),
                TypeVar("Small", bound=Range[L[1], L[4]]),
            ),
        )
    ),
    "pep695": dict(zip(FORMS, _Declared.__type_params__)),
}


@pytest.mark.parametrize("syntax", list(TYPEVARS))
@pytest.mark.parametrize(
    "form,shape,valid",
    [
        pytest.param("int bound", (3, 3), True, id="int-agree"),
        pytest.param("int bound", (3, 4), False, id="int-disagree"),
        pytest.param("literal constraints", (4, 4), True, id="constraint-agree"),
        pytest.param("literal constraints", (3, 4), False, id="constraint-disagree"),
        pytest.param("literal constraints", (5, 5), False, id="constraint-outside"),
        pytest.param("union of literal constraints", (5, 5), True, id="union-agree"),
        pytest.param(
            "union of literal constraints", (2, 3), False, id="union-same-group"
        ),
        pytest.param("union of literal constraints", (6, 6), False, id="union-outside"),
        pytest.param("range bound", (4, 4), True, id="range-agree"),
        pytest.param("range bound", (3, 4), False, id="range-disagree"),
        pytest.param("range bound", (5, 5), False, id="range-outside"),
    ],
)
def test_within_array(syntax: str, form: str, shape: tuple, valid: bool) -> None:
    """Each kind of shape typevar is matched within an array the same way"""
    var = TYPEVARS[syntax][form]
    assert isinstance(np.zeros(shape), NDArray[Shape[var, var], Any]) is valid


# the TypeVar syntax, to compare with the 3.12 syntax below
class TypeVarModel(BaseModel, Generic[N, K]):  # noqa: UP046
    model_config = ConfigDict(validate_assignment=True)

    vector: NDArray[Shape[N], F]
    matrix: NDArray[Shape[N, K], F]


class Pep695Model[N: int, K: int](BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    vector: NDArray[Shape[N], F]
    matrix: NDArray[Shape[N, K], F]


MODELS = {"TypeVar": TypeVarModel, "pep695": Pep695Model}


@pytest.mark.parametrize("syntax", list(MODELS))
def test_model_fields(syntax: str) -> None:
    """The fields of a generic model have to agree"""
    model = MODELS[syntax]
    model(vector=np.zeros(3), matrix=np.zeros((3, 4)))
    with pytest.raises(ValidationError):
        model(vector=np.zeros(3), matrix=np.zeros((4, 4)))


@pytest.mark.parametrize("syntax", list(MODELS))
def test_assignment(syntax: str) -> None:
    """Sizes are fixed once a model is made"""
    instance = MODELS[syntax](vector=np.zeros(3), matrix=np.zeros((3, 4)))
    instance.vector = np.zeros(3)
    with pytest.raises(ValidationError):
        instance.vector = np.zeros(4)


@pytest.mark.parametrize("syntax", list(MODELS))
def test_parametrized(syntax: str) -> None:
    """Parametrizing a model replaces its shape typevars"""
    exact = MODELS[syntax][L[3], L[4]]
    exact(vector=np.zeros(3), matrix=np.zeros((3, 4)))
    with pytest.raises(ValidationError):
        exact(vector=np.zeros(5), matrix=np.zeros((5, 4)))
