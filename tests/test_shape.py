import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated as A
from typing import Any, Literal
from typing import Literal as L

import numpy as np
import pytest
from annotated_types import Interval
from pydantic import BaseModel, ValidationError

from numpydantic import AnyDims, NDArray, Range, Shape
from numpydantic.exceptions import InvalidShapeExpressionError, ShapeError

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack

pytestmark = pytest.mark.shape

STRING_DEPRECATION = "String shape expressions"

ShapeTuple = tuple[int, ...]


@dataclass
class Form:
    """One way of writing a shape feature"""

    id: str
    build: Callable[[], Any]
    """Builds the annotation lazily, so unsupported forms fail one test, not import"""
    deprecated: bool = False
    """Whether this form should emit a deprecation warning for using strings"""
    todo: str | None = None
    todo_shapes: list[ShapeTuple] | None = None
    """Only expect these array shapes to fail, rather than all of them"""
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
        "one-dim",
        valid=[(3,)],
        invalid=[(4,), (3, 3), ()],
        forms=[
            Form("str", lambda: Shape["3"], deprecated=True),
            Form("literal-str-expr", lambda: Shape[L["3"]], deprecated=True),
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
            Form("str", lambda: Shape["3, 4"], deprecated=True),
            Form("literal-str", lambda: Literal["3, 4"], deprecated=True),
            Form("call-str", lambda: Shape("3, 4"), deprecated=True),
            Form("args", lambda: Shape[3, 4]),
            Form("call", lambda: Shape(3, 4)),
            Form("tuple-args", lambda: tuple[3, 4]),
            Form("literal", lambda: Shape[L[3], L[4]]),
            Form("tuple", lambda: tuple[L[3], L[4]]),
        ],
        canonical=tuple[L[3], L[4]],
    ),
    ShapeFeature(
        "wildcard",
        valid=[(1, 1), (5, 9)],
        invalid=[(3,), (3, 3, 3)],
        forms=[
            Form("str", lambda: Shape["*, *"], deprecated=True),
            Form("args", lambda: Shape["*", "*"], deprecated=True),
            Form("literal-str-dims", lambda: Shape[L["*"], L["*"]], deprecated=True),
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
            Form("str", lambda: Shape["2-4, 6-7"], deprecated=True),
            Form("args", lambda: Shape["2-4", "6-7"], deprecated=True),
            Form(
                "literal-str-dims",
                lambda: Shape[L["2-4"], L["6-7"]],
                deprecated=True,
            ),
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
            Form("str", lambda: Shape["2-*, *-5"], deprecated=True),
            Form("args", lambda: Shape["2-*", "*-5"], deprecated=True),
            Form(
                "literal-str-dims",
                lambda: Shape[L["2-*"], L["*-5"]],
                deprecated=True,
            ),
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
            Form("str", lambda: Shape["1, 2, ..."], deprecated=True),
            Form("literal-str", lambda: Literal["1, 2, ..."], deprecated=True),
            Form("call-str", lambda: Shape("1, 2, ..."), deprecated=True),
            Form("args", lambda: Shape[1, 2, "..."], deprecated=True),
            Form("call", lambda: Shape(1, 2, "..."), deprecated=True),
            Form(
                "literal-str-dims",
                lambda: Shape[L[1], L[2], L["..."]],
                deprecated=True,
            ),
            Form("anydims", lambda: Shape[L[1], L[2], Unpack[AnyDims]]),
            Form(
                "anydims-star",
                lambda: Shape[(L[1], L[2], *AnyDims)],
                min_version=(3, 11),
            ),
            Form("tuple-anydims", lambda: tuple[L[1], L[2], Unpack[AnyDims]]),
        ],
        canonical=tuple[L[1], L[2], *tuple[int, ...]],
    ),
    ShapeFeature(
        "one-then-variadic",
        valid=[(2,), (2, 3), (2, 3, 4)],
        invalid=[(), (3,), (3, 2)],
        forms=[
            Form("str", lambda: Shape["2, ..."], deprecated=True),
            Form("args", lambda: Shape[2, "..."], deprecated=True),
            Form("anydims", lambda: Shape[L[2], Unpack[AnyDims]]),
            Form(
                "anydims-star",
                lambda: Shape[(L[2], *AnyDims)],
                min_version=(3, 11),
            ),
            Form("tuple-anydims", lambda: tuple[L[2], Unpack[AnyDims]]),
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
            Form("str", lambda: Shape["*, ..."], deprecated=True),
            Form("int", lambda: Shape[int, ...]),
            Form("tuple", lambda: tuple[int, ...]),
            Form("anydims", lambda: Shape[Unpack[AnyDims]]),
            Form("anydims-bare", lambda: AnyDims),
        ],
        canonical=Any,
    ),
    ShapeFeature(
        "labels",
        valid=[(3, 7)],
        invalid=[(4, 7)],
        forms=[Form("str", lambda: Shape["3 x, * y"], deprecated=True)],
        canonical=tuple[A[L[3], "x"], A[int, "y"]],
    ),
]


def _shape_id(shape: ShapeTuple) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "0d"


def _form_marks(form: Form, shape: ShapeTuple | None = None) -> list:
    marks = []
    if form.todo and (form.todo_shapes is None or shape in form.todo_shapes):
        marks.append(pytest.mark.xfail(strict=True, reason=form.todo))
    if form.min_version:
        marks.append(
            pytest.mark.skipif(
                sys.version_info < form.min_version,
                reason=f"requires python {form.min_version}",
            )
        )
    return marks


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
                        marks=_form_marks(form, shape),
                    )
                    for shape in shapes
                )
    return params


def _form_params(features: list[ShapeFeature]) -> list:
    return [
        pytest.param(
            form,
            feature.canonical,
            id=f"{feature.id}-{form.id}",
            marks=_form_marks(form),
        )
        for feature in features
        for form in feature.forms
    ]


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


@pytest.mark.parametrize("form, canonical", _form_params(SHAPE_FEATURES))
def test_shape_form_deprecation(form: Form, canonical: type) -> None:
    """Shapes written with strings warn that they are deprecated, and nothing else"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = NDArray[form.build(), Any]

    warned = any(
        issubclass(w.category, DeprecationWarning)
        and str(w.message).startswith(STRING_DEPRECATION)
        for w in caught
    )
    assert warned == form.deprecated


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
        pytest.param(lambda: Shape[Unpack[AnyDims], L[3]], id="anydims-not-last"),
        pytest.param(lambda: Shape[...], id="ellipsis-alone"),
        pytest.param(lambda: Shape[L[2], ..., L[3]], id="ellipsis-not-last"),
    ],
)
def test_shape_invalid_forms(build: Callable[[], Any]) -> None:
    """Misused Range, AnyDims, and ... raise a clear error when the shape is declared"""
    with pytest.raises(InvalidShapeExpressionError):
        _ = NDArray[build(), Any]


@pytest.mark.parametrize("form,canonical", _form_params(SHAPE_FEATURES))
def test_shape_canonical_form(form: Form, canonical: type) -> None:
    """Every shape should be equivalent to the expected canonical form"""
    ndarray = NDArray[form.build(), Any]
    assert ndarray.__args__[0] == canonical


def test_range_shape_schema():
    """
    Range shapes should correctly generate JSON Schema
    """

    class MyModel(BaseModel):
        array_range: NDArray[Shape["2-4"], Any]
        array_range_min: NDArray[Shape["2-*"], Any]
        array_range_max: NDArray[Shape["*-4"], Any]

    schema = MyModel.model_json_schema()
    assert schema["properties"]["array_range"]["minItems"] == 2
    assert schema["properties"]["array_range"]["maxItems"] == 4
    assert schema["properties"]["array_range_min"]["minItems"] == 2
    assert "maxItems" not in schema["properties"]["array_range_min"]
    assert schema["properties"]["array_range_max"]["maxItems"] == 4
    assert "minItems" not in schema["properties"]["array_range_max"]
