"""
Shape typevars within one array, and shared between the arrays of a model
"""

import dataclasses
import sys
from pathlib import Path
from typing import Annotated as A
from typing import Any, Generic, TypeVar
from typing import Literal as L

import numpy as np
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    TypeAdapter,
    ValidationError,
    validate_call,
)
from pydantic.dataclasses import dataclass
from pydantic_core import core_schema

from numpydantic import NDArray, Shape
from numpydantic.testing.cases import TYPEVAR_CASES
from numpydantic.testing.helpers import ValidationCase
from numpydantic.testing.interfaces import NumpyCase

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

pytestmark = pytest.mark.shape


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in TYPEVAR_CASES])
def test_typevar_cases(case: ValidationCase, tmp_path: Path) -> None:
    """Shape typevars are matched within one array, the same way for every interface"""
    case.model_copy(update={"interface": NumpyCase}).validate_case(tmp_path)


N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


class Model(BaseModel, Generic[N, K]):
    model_config = ConfigDict(validate_assignment=True)

    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, K], np.float64]
    cube: NDArray[Shape[L[1], K, ...], np.float64] | None = None
    hypercube: NDArray[Shape[A[N, "side"], ...], np.float64] | None = None
    vectors: list[NDArray[Shape[N], np.float64]] = []


def _model(n: int = 3, k: int = 4, **kwargs: Any) -> Model:
    return Model(vector=np.zeros(n), matrix=np.zeros((n, k)), **kwargs)


def test_fields_agree() -> None:
    """Arrays in different fields can have any sizes, as long as they agree"""
    _model(
        3,
        4,
        cube=np.zeros((1, 4, 4)),
        hypercube=np.zeros((3, 3, 3)),
        vectors=[np.zeros(3)],
    )
    _model(5, 2, cube=np.zeros((1, 2, 2, 2)))


@pytest.mark.parametrize(
    "kwargs,loc",
    [
        pytest.param({"matrix": np.zeros((4, 4))}, ("matrix",), id="first-dim"),
        pytest.param({"cube": np.zeros((1, 4, 3))}, ("cube",), id="repeated"),
        pytest.param({"hypercube": np.zeros((4, 4))}, ("hypercube",), id="labelled"),
        pytest.param({"vectors": [np.zeros(4)]}, ("vectors", 0), id="in-list"),
    ],
)
def test_fields_disagree(kwargs: dict[str, Any], loc: tuple) -> None:
    """A field whose sizes disagree with an earlier field is invalid"""
    fields = {"vector": np.zeros(3), "matrix": np.zeros((3, 4)), **kwargs}
    with pytest.raises(ValidationError) as e:
        Model(**fields)
    assert e.value.errors()[0]["loc"] == loc


class ListOnly(BaseModel, Generic[N]):
    vectors: list[NDArray[Shape[N], np.float64]]


class ListFirst(BaseModel, Generic[N]):
    vectors: list[NDArray[Shape[N], np.float64]]
    vector: NDArray[Shape[N], np.float64]


def test_list_items() -> None:
    """Items of a list agree with each other and with the fields after the list"""
    ListOnly(vectors=[np.zeros(3), np.zeros(3)])
    ListFirst(vectors=[np.zeros(3)], vector=np.zeros(3))
    with pytest.raises(ValidationError):
        ListOnly(vectors=[np.zeros(3), np.zeros(4)])
    with pytest.raises(ValidationError):
        ListFirst(vectors=[np.zeros(3)], vector=np.zeros(4))


def test_disagree_message() -> None:
    """The error says which sizes were already set"""
    with pytest.raises(ValidationError, match="N=3"):
        Model(vector=np.zeros(3), matrix=np.zeros((4, 4)))


def test_assignment() -> None:
    """
    Sizes are fixed once a model is made, as they are for a type checker,
    so assigned arrays have to have the same sizes
    """
    model = _model(3, 4)
    model.vector = np.zeros(3)
    with pytest.raises(ValidationError):
        model.vector = np.zeros(4)
    with pytest.raises(ValidationError):
        model.matrix = np.zeros((3, 5))


class Partial(BaseModel, Generic[N, K]):
    model_config = ConfigDict(validate_assignment=True)

    vector: NDArray[Shape[N], np.float64]
    other: NDArray[Shape[K], np.float64] | None = None


def test_assignment_sets_size() -> None:
    """A size first set by an assignment is kept for later assignments"""
    model = Partial(vector=np.zeros(3))
    model.other = np.zeros(5)
    with pytest.raises(ValidationError):
        model.other = np.zeros(6)


class SharesK(BaseModel, Generic[N, K]):
    model_config = ConfigDict(validate_assignment=True)

    vector: NDArray[Shape[K], np.float64]
    matrix: NDArray[Shape[N, K], np.float64] | None = None


def test_refused_assignment_sets_no_size() -> None:
    """
    A refused assignment doesn't keep sizes from the part of the array that matched.

    The assigned array matches N before its K is refused,
    and the model keeps its sizes in the same dict that assignment validates against.
    """
    model = SharesK(vector=np.zeros(7))
    with pytest.raises(ValidationError):
        model.matrix = np.zeros((5, 6))
    model.matrix = np.zeros((4, 7))


def test_sizes_hidden() -> None:
    """The stored sizes aren't dumped, and don't change equality"""
    model = _model(3, 4)
    assert "__npd_typevars__" in model.__dict__
    assert "__npd_typevars__" not in model.model_dump()
    assert model == Model.model_construct(**model.model_dump())


def test_parametrized() -> None:
    """Parametrizing a model replaces its shape typevars"""
    Exact = Model[L[3], L[4]]
    assert Exact.model_fields["matrix"].annotation.__args__[0] == tuple[L[3], L[4]]
    Exact(vector=np.zeros(3), matrix=np.zeros((3, 4)))
    with pytest.raises(ValidationError):
        Exact(vector=np.zeros(5), matrix=np.zeros((5, 4)))


def test_scoped_to_model() -> None:
    """Models that use the same type variables don't share sizes"""

    class Inner(BaseModel, Generic[N]):
        vector: NDArray[Shape[N], np.float64]

    class Outer(BaseModel, Generic[N]):
        vector: NDArray[Shape[N], np.float64]
        inner: Inner

    Outer(vector=np.zeros(3), inner=Inner(vector=np.zeros(4)))
    Outer(vector=np.zeros(3), inner={"vector": np.zeros(4)})


class Containers(BaseModel, Generic[N]):
    vector: NDArray[Shape[N], np.float64]
    by_name: dict[str, NDArray[Shape[N], np.float64]] = {}
    pair: (
        tuple[NDArray[Shape[N], np.float64], NDArray[Shape[N, N], np.float64]] | None
    ) = None


@pytest.mark.parametrize(
    "kwargs,valid",
    [
        pytest.param({"by_name": {"a": np.zeros(3)}}, True, id="dict-agree"),
        pytest.param({"by_name": {"a": np.zeros(4)}}, False, id="dict-disagree"),
        pytest.param({"pair": (np.zeros(3), np.zeros((3, 3)))}, True, id="tuple-agree"),
        pytest.param(
            {"pair": (np.zeros(3), np.zeros((4, 4)))}, False, id="tuple-disagree"
        ),
    ],
)
def test_containers(kwargs: dict[str, Any], valid: bool) -> None:
    """Arrays in dicts and tuples agree with the other fields"""
    if valid:
        Containers(vector=np.zeros(3), **kwargs)
    else:
        with pytest.raises(ValidationError):
            Containers(vector=np.zeros(3), **kwargs)


@dataclass
class DataclassFields(Generic[N]):
    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, N], np.float64]


def test_dataclass() -> None:
    """The fields of a pydantic dataclass agree, like a model's"""
    DataclassFields(vector=np.zeros(3), matrix=np.zeros((3, 3)))
    with pytest.raises(ValidationError):
        DataclassFields(vector=np.zeros(3), matrix=np.zeros((4, 4)))


@dataclasses.dataclass
class StdlibDataclassFields:
    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, N], np.float64]


class HoldsStdlibDataclass(BaseModel):
    fields: StdlibDataclassFields


def test_stdlib_dataclass() -> None:
    """The fields of a standard library dataclass agree when pydantic validates it"""
    HoldsStdlibDataclass(fields={"vector": np.zeros(3), "matrix": np.zeros((3, 3))})
    with pytest.raises(ValidationError):
        HoldsStdlibDataclass(fields={"vector": np.zeros(3), "matrix": np.zeros((4, 4))})


class TypedDictFields(TypedDict):
    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, N], np.float64]


def test_typeddict() -> None:
    """
    A validated TypedDict is the dict that pydantic returns,
    so no sizes are kept in it, and shape typevars are matched within each array
    """
    adapter = TypeAdapter(TypedDictFields)
    validated = adapter.validate_python(
        {"vector": np.zeros(3), "matrix": np.zeros((3, 3))}
    )
    assert list(validated) == ["vector", "matrix"]
    with pytest.raises(ValidationError):
        adapter.validate_python({"vector": np.zeros(3), "matrix": np.zeros((3, 4))})


def test_owner_known() -> None:
    """
    pydantic only says which model a field belongs to through a private attribute,
    ``handler._generate_schema.model_type_stack``.
    This fails if a pydantic version removes it or it no longer gives the model,
    which turns off matching shape typevars between fields.
    """
    owners = []

    class Spy:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            # purposeful access of private variable, the test is that we can do that
            stack = handler._generate_schema.model_type_stack
            owners.append(stack.get())
            return core_schema.any_schema()

    class Owner(BaseModel):
        field: Spy

    assert owners == [Owner]


def test_validate_call() -> None:
    """Shape typevars are matched within an argument of a validated function"""

    @validate_call
    def trace(matrix: NDArray[Shape[N, N], np.float64]) -> float:
        return float(np.trace(matrix))

    trace(np.zeros((3, 3)))
    with pytest.raises(ValidationError):
        trace(np.zeros((3, 4)))
