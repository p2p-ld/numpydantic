import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated as A
from typing import Any, Literal, TypeVarTuple, Unpack
from typing import Literal as L

import numpy as np
import pytest
from annotated_types import Interval
from pydantic import BaseModel, ValidationError

from numpydantic import AnyDims, NDArray, Range, Shape, transposed, unshape
from numpydantic.exceptions import InvalidShapeExpressionError, ShapeError

pytestmark = pytest.mark.shape

STRING_DEPRECATION = "String shape expressions"

ShapeTuple = tuple[int, ...]


@dataclass
class Form:
    """One way of writing a shape feature"""

    id: str
    build: Callable[[], Any]
    """Builds the annotation lazily, so unsupported forms fail one test, not import"""
    min_version: tuple[int, int] | None = None


@dataclass
class ShapeFeature:
    """A shape constraint, the arrays it should accept and reject, and its forms"""

    id: str
    valid: list[ShapeTuple]
    canonical: type[tuple]
    invalid: list[ShapeTuple] = field(default_factory=list)
    forms: list[Form] = field(default_factory=list)


SHAPE_FEATURES = [
    ShapeFeature(
        "zero-dim",
        valid=[()],
        invalid=[(1,), (1, 2)],
        forms=[
            Form("empty", lambda: Shape[()]),
            Form("tuple", lambda: tuple[()]),
        ],
        canonical=tuple[()],
    ),
    ShapeFeature(
        "one-dim",
        valid=[(3,)],
        invalid=[(4,), (3, 3), ()],
        forms=[
            Form("args", lambda: Shape[3]),
            Form("literal", lambda: Shape[L[3]]),
            Form("tuple", lambda: tuple[L[3]]),
        ],
        canonical=tuple[L[3]],
    ),
    ShapeFeature(
        "exact",
        valid=[(3, 4)],
        invalid=[(4, 3), (3,), (3, 4, 1)],
        forms=[
            Form("args", lambda: Shape[3, 4]),
            Form("call", lambda: Shape(3, 4)),
            Form("tuple-args", lambda: tuple[3, 4]),
            Form("literal", lambda: Shape[L[3], L[4]]),
            Form("tuple", lambda: tuple[L[3], L[4]]),
            Form("unpack-fixed", lambda: Shape[L[3], *tuple[L[4]]]),
            Form(
                "unpack-fixed-nested",
                lambda: Shape[*tuple[L[3], *tuple[L[4]]]],
            ),
            Form("tuple-unpack-fixed", lambda: tuple[L[3], *tuple[L[4]]]),
            Form(
                "typing-unpack-fixed",
                lambda: Shape[L[3], Unpack[tuple[L[4]]]],  # noqa: UP044
            ),
        ],
        canonical=tuple[L[3], L[4]],
    ),
    ShapeFeature(
        "wildcard",
        valid=[(1, 1), (5, 9)],
        invalid=[(3,), (3, 3, 3)],
        forms=[
            Form("int", lambda: Shape[int, int]),
            Form("tuple", lambda: tuple[int, int]),
        ],
        canonical=tuple[int, int],
    ),
    ShapeFeature(
        "range",
        valid=[(2, 6), (2, 7), (3, 6), (3, 7), (4, 6), (4, 7)],
        invalid=[(1, 6), (5, 6), (2, 5), (2, 8)],
        forms=[
            Form("range", lambda: Shape[Range[L[2], L[4]], Range[L[6], L[7]]]),
            Form("tuple", lambda: tuple[Range[L[2], L[4]], Range[L[6], L[7]]]),
        ],
        canonical=tuple[
            A[int, Interval(ge=2, le=4)],
            A[int, Interval(ge=6, le=7)],
        ],
    ),
    ShapeFeature(
        "open-range",
        valid=[(2, 5), (10, 5), (2, 2)],
        invalid=[(1, 5), (2, 6)],
        forms=[
            Form("range", lambda: Shape[Range[L[2], int], Range[int, L[5]]]),
        ],
        canonical=tuple[
            A[int, Interval(ge=2)],
            A[int, Interval(le=5)],
        ],
    ),
    ShapeFeature(
        "trailing-variadic",
        valid=[(1, 2), (1, 2, 3), (1, 2, 3, 4)],
        invalid=[(4, 5), (1,), (2, 1, 3), ()],
        forms=[
            Form("anydims", lambda: Shape[L[1], L[2], *AnyDims]),
            Form(
                "anydims-star",
                lambda: Shape[(L[1], L[2], *AnyDims)],
                min_version=(3, 11),
            ),
            Form("tuple-anydims", lambda: tuple[L[1], L[2], *AnyDims]),
            Form(
                "anydims-nested",
                lambda: Shape[L[1], *tuple[L[2], *AnyDims]],
            ),
            Form(
                "typing-unpack-anydims",
                lambda: Shape[L[1], L[2], Unpack[AnyDims]],  # noqa: UP044
            ),
        ],
        canonical=tuple[L[1], L[2], *tuple[int, ...]],
    ),
    ShapeFeature(
        "one-then-variadic",
        valid=[(2,), (2, 3), (2, 3, 4)],
        invalid=[(), (3,), (3, 2)],
        forms=[
            Form("anydims", lambda: Shape[L[2], *AnyDims]),
            Form(
                "anydims-star",
                lambda: Shape[(L[2], *AnyDims)],
                min_version=(3, 11),
            ),
            Form("tuple-anydims", lambda: tuple[L[2], *AnyDims]),
        ],
        canonical=tuple[L[2], *AnyDims],
    ),
    ShapeFeature(
        "repeat-last",
        valid=[(1, 2), (1, 2, 2), (1, 2, 2, 2)],
        invalid=[(), (1,), (1, 3), (1, 2, 3)],
        forms=[
            Form("args", lambda: Shape[1, 2, ...]),
            Form("literal", lambda: Shape[L[1], L[2], ...]),
            Form("tuple", lambda: tuple[L[1], L[2], ...]),
            Form("unpack", lambda: Shape[L[1], L[2], *tuple[L[2], ...]]),
            Form("tuple-unpack", lambda: tuple[L[1], L[2], *tuple[L[2], ...]]),
        ],
        canonical=tuple[L[1], L[2], *tuple[L[2], ...]],
    ),
    ShapeFeature(
        "homogeneous",
        valid=[(), (2,), (2, 2, 2)],
        invalid=[(3,), (2, 3)],
        forms=[
            Form("args", lambda: Shape[2, ...]),
            Form("literal", lambda: Shape[L[2], ...]),
            Form("tuple", lambda: tuple[L[2], ...]),
        ],
        canonical=tuple[L[2], ...],
    ),
    ShapeFeature(
        "homogeneous-range",
        valid=[(), (2,), (2, 3, 2)],
        invalid=[(4,), (2, 4)],
        forms=[
            Form("range", lambda: Shape[Range[L[2], L[3]], ...]),
            Form("tuple", lambda: tuple[Range[L[2], L[3]], ...]),
        ],
        canonical=tuple[A[int, Interval(ge=2, le=3)], ...],
    ),
    ShapeFeature(
        "any",
        valid=[(), (3,), (1, 2, 3)],
        forms=[
            Form("any", lambda: Any),
            Form("bare", lambda: Shape),
            Form("int", lambda: Shape[int, ...]),
            Form("tuple", lambda: tuple[int, ...]),
            Form("anydims", lambda: Shape[*AnyDims]),
            Form("anydims-bare", lambda: AnyDims),
        ],
        canonical=Any,
    ),
    ShapeFeature(
        "labels",
        valid=[(3, 7)],
        invalid=[(4, 7)],
        forms=[
            Form("annotated", lambda: Shape[A[L[3], "x"], A[int, "y"]]),
            Form("tuple", lambda: tuple[A[L[3], "x"], A[int, "y"]]),
        ],
        canonical=tuple[A[L[3], "x"], A[int, "y"]],
    ),
]

_CANONICAL = {feature.id: feature.canonical for feature in SHAPE_FEATURES}

STRING_FORMS = {
    "one-dim": [
        Form("str", lambda: Shape["3"]),
        Form("literal-str", lambda: Shape[L["3"]]),
    ],
    "exact": [
        Form("str", lambda: Shape["3, 4"]),
        Form("literal-str", lambda: Literal["3, 4"]),
        Form("shape-literal-str", lambda: Shape[L["3, 4"]]),
        Form("call-str", lambda: Shape("3, 4")),
    ],
    "wildcard": [
        Form("str", lambda: Shape["*, *"]),
        Form("args", lambda: Shape["*", "*"]),
        Form("literal-str-dims", lambda: Shape[L["*"], L["*"]]),
    ],
    "range": [
        Form("str", lambda: Shape["2-4, 6-7"]),
        Form("args", lambda: Shape["2-4", "6-7"]),
        Form("literal-str-dims", lambda: Shape[L["2-4"], L["6-7"]]),
        Form("shape-literal-str", lambda: Shape[L["2-4, 6-7"]]),
    ],
    "open-range": [
        Form("str", lambda: Shape["2-*, *-5"]),
        Form("args", lambda: Shape["2-*", "*-5"]),
        Form("literal-str-dims", lambda: Shape[L["2-*"], L["*-5"]]),
    ],
    "trailing-variadic": [
        Form("str", lambda: Shape["1, 2, ..."]),
        Form("literal-str", lambda: Literal["1, 2, ..."]),
        Form("call-str", lambda: Shape("1, 2, ...")),
        Form("args", lambda: Shape[1, 2, "..."]),
        Form("call", lambda: Shape(1, 2, "...")),
        Form("literal-str-dims", lambda: Shape[L[1], L[2], L["..."]]),
    ],
    "one-then-variadic": [
        Form("str", lambda: Shape["2, ..."]),
        Form("args", lambda: Shape[2, "..."]),
    ],
    "any": [Form("str", lambda: Shape["*, ..."])],
    "labels": [Form("str", lambda: Shape["3 x, * y"])],
}


def _shape_id(shape: ShapeTuple) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "0d"


def _form_marks(form: Form) -> list:
    if form.min_version:
        return [
            pytest.mark.skipif(
                sys.version_info < form.min_version,
                reason=f"requires python {form.min_version}",
            )
        ]
    return []


def _shape_params(features: list[ShapeFeature]) -> list:
    params = []
    for feature in features:
        for form in feature.forms:
            for shapes, valid, label in (
                (feature.valid, True, "valid"),
                (feature.invalid, False, "invalid"),
            ):
                params.extend(
                    pytest.param(
                        form.build,
                        shape,
                        valid,
                        id=f"{feature.id}-{form.id}-{_shape_id(shape)}-{label}",
                        marks=_form_marks(form),
                    )
                    for shape in shapes
                )
    return params


def _form_params(features: list[ShapeFeature]) -> list:
    return [
        pytest.param(
            form.build,
            feature.canonical,
            id=f"{feature.id}-{form.id}",
            marks=_form_marks(form),
        )
        for feature in features
        for form in feature.forms
    ]


def _string_form_params() -> list:
    params = [
        pytest.param(form.build, _CANONICAL[feature_id], id=f"{feature_id}-{form.id}")
        for feature_id, forms in STRING_FORMS.items()
        for form in forms
    ]
    params.append(
        pytest.param(
            lambda: Shape["2-4 x"],
            tuple[A[int, Interval(ge=2, le=4), "x"]],
            id="labelled-range-str",
        )
    )
    return params


@pytest.mark.parametrize("build,shape,valid", _shape_params(SHAPE_FEATURES))
def test_shape_forms(build: Callable[[], Any], shape: ShapeTuple, valid: bool) -> None:
    """
    Every form of a shape feature accepts and rejects the same arrays,
    both when calling NDArray directly and as a pydantic field.
    """
    array_type = NDArray[build(), Any]

    class MyModel(BaseModel):
        array: array_type

    array = np.zeros(shape)
    if valid:
        _ = array_type(array)
        _ = MyModel(array=array)
    else:
        with pytest.raises(ShapeError):
            _ = array_type(array)
        with pytest.raises(ValidationError):
            _ = MyModel(array=array)


@pytest.mark.parametrize("build,canonical", _form_params(SHAPE_FEATURES))
def test_shape_canonical_form(build: Callable[[], Any], canonical: type) -> None:
    """Every form of a shape feature normalizes to the same canonical form"""
    ndarray = NDArray[build(), Any]
    assert ndarray.__args__[0] == canonical


@pytest.mark.parametrize("build,canonical", _form_params(SHAPE_FEATURES))
def test_shape_form_no_deprecation(build: Callable[[], Any], canonical: type) -> None:
    """Shapes that aren't written with strings aren't deprecated"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = NDArray[build(), Any]

    assert not any(
        issubclass(w.category, DeprecationWarning)
        and str(w.message).startswith(STRING_DEPRECATION)
        for w in caught
    )


@pytest.mark.parametrize("build,canonical", _string_form_params())
def test_string_form(build: Callable[[], Any], canonical: type) -> None:
    """
    The deprecated string forms still work, and warn,
    by normalizing to the same canonical form as the type expression they replace
    """
    with pytest.warns(DeprecationWarning, match=STRING_DEPRECATION):
        ndarray = NDArray[build(), Any]
    assert ndarray.__args__[0] == canonical


def test_string_ellipsis_deprecation() -> None:
    """
    Strings with '...' also warn about its changed meaning,
    pointing to the line where the shape was declared
    """
    with pytest.warns(DeprecationWarning, match=r"now written as \*AnyDims") as caught:
        _ = NDArray[Literal["1, 2, ..."], Any]
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Range[L[2]], id="range-one-bound"),
        pytest.param(lambda: Range[2, 4], id="range-bare-int"),
        pytest.param(lambda: Range[L[4], L[2]], id="range-low-over-high"),
        pytest.param(lambda: Range[L["2"], L[4]], id="range-str-bound"),
        pytest.param(lambda: Range[L[-1], L[4]], id="range-negative-bound"),
        pytest.param(lambda: Range[L[2, 3], L[4]], id="range-multi-literal"),
        pytest.param(lambda: Shape[L[3, 4]], id="multi-literal"),
        pytest.param(lambda: Shape[*AnyDims, L[3]], id="anydims-not-last"),
        pytest.param(lambda: Shape[...], id="ellipsis-alone"),
        pytest.param(lambda: Shape[L[2], ..., L[3]], id="ellipsis-not-last"),
        pytest.param(
            lambda: Shape[*tuple[*AnyDims, L[3]]],
            id="anydims-not-last-after-flattening",
        ),
        pytest.param(lambda: Shape["...", 3], id="string-ellipsis-not-last"),
        pytest.param(lambda: Shape["1, 2", 3], id="string-dimension-with-comma"),
        pytest.param(lambda: Shape["Dim, Dim"], id="string-variables"),
    ],
)
@pytest.mark.filterwarnings(f"ignore:{STRING_DEPRECATION}:DeprecationWarning")
def test_shape_invalid_forms(build: Callable[[], Any]) -> None:
    """Misused Range, AnyDims, and ... raise a clear error when the shape is declared"""
    with pytest.raises(InvalidShapeExpressionError):
        _ = NDArray[build(), Any]


def test_typevartuple_passthrough() -> None:
    """Shape TypeVarTuples aren't supported yet, so they're kept as they are"""
    Ts = TypeVarTuple("Ts")
    assert Shape[L[3], *Ts] == tuple[L[3], *Ts]
    assert Shape[*Ts, L[3]] == tuple[*Ts, L[3]]


def test_range_shape_schema():
    """
    Range shapes should correctly generate JSON Schema
    """

    class MyModel(BaseModel):
        array_range: NDArray[Shape[Range[L[2], L[4]]], Any]
        array_range_min: NDArray[Shape[Range[L[2], int]], Any]
        array_range_max: NDArray[Shape[Range[int, L[4]]], Any]
        array_range_zero: NDArray[Shape[Range[L[0], L[4]]], Any]
        array_range_exclusive: NDArray[Shape[A[int, Interval(gt=1, lt=5)]], Any]

    schema = MyModel.model_json_schema()["properties"]
    assert schema["array_range"]["minItems"] == 2
    assert schema["array_range"]["maxItems"] == 4
    assert schema["array_range_min"]["minItems"] == 2
    assert "maxItems" not in schema["array_range_min"]
    assert schema["array_range_max"]["maxItems"] == 4
    assert "minItems" not in schema["array_range_max"]
    assert schema["array_range_zero"]["minItems"] == 0
    assert schema["array_range_exclusive"]["minItems"] == 2
    assert schema["array_range_exclusive"]["maxItems"] == 4


def test_transposed() -> None:
    """transposed does nothing at runtime"""
    array = np.zeros((3, 4)).T
    assert transposed(array) is array


def test_unshape() -> None:
    """unshape does nothing at runtime"""
    array = np.zeros((3, 4))
    assert unshape(array) is array
