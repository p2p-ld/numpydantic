from typing import Any, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from numpydantic import NDArray, Shape
from numpydantic.exceptions import ShapeError

pytestmark = pytest.mark.shape


@pytest.mark.parametrize(
    "shape,valid",
    [
        ((2, 6), True),
        ((2, 7), True),
        ((3, 6), True),
        ((3, 7), True),
        ((4, 6), True),
        ((4, 7), True),
        ((1, 6), False),
        ((5, 6), False),
        ((2, 5), False),
        ((2, 8), False),
    ],
)
def test_shape_range(shape, valid):
    """Specify a dimension with a range of possible sizes"""

    class MyModel(BaseModel):
        array: NDArray[Shape["2-4, 6-7"], Any]

    if valid:
        _ = MyModel(array=np.zeros(shape, dtype=np.uint8))
    else:
        with pytest.raises(ValidationError):
            _ = MyModel(array=np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize(
    "shape,valid",
    [
        ((2, 5), True),
        ((10, 5), True),
        ((2, 2), True),
        ((1, 5), False),
        ((2, 6), False),
    ],
)
def test_shape_wildcard(shape, valid):
    """Specify an open-ended minimum or maximum size for a given dimension"""

    class MyModel(BaseModel):
        array: NDArray[Shape["2-*, *-5"], Any]

    if valid:
        _ = MyModel(array=np.zeros(shape, dtype=np.uint8))
    else:
        with pytest.raises(ValidationError):
            _ = MyModel(array=np.zeros(shape, dtype=np.uint8))


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


def test_shape_literal():
    """
    We can use `Literal` instead of the `Shape` object.

    We do not test for correctness of validation here,
    assuming that it handles `Literal` strings exactly the same
    way that it does `Shape[]` strings.
    """
    array_type = NDArray[Literal["1, 2, ..."], Any]

    # validates
    _ = array_type(np.zeros((1, 2, 3)))
    # fails to validate
    with pytest.raises(ShapeError):
        _ = array_type(np.zeros((4, 5)))


def test_shape_callable():
    """Shape can be used as a callable rather than a bracketed type"""
    array_type = NDArray[Shape("1, 2, ..."), Any]

    # validates
    _ = array_type(np.zeros((1, 2, 3)))
    # fails to validate
    with pytest.raises(ShapeError):
        _ = array_type(np.zeros((4, 5)))


@pytest.mark.parametrize("form", ["square", "round"])
def test_shape_args(form: str):
    """Shape can be given as args as well!"""
    if form == "square":
        array_type = NDArray[Shape[1, 2, "..."], Any]
    else:
        array_type = NDArray[Shape(1, 2, "..."), Any]

    # validates
    _ = array_type(np.zeros((1, 2, 3)))
    # fails to validate
    with pytest.raises(ShapeError):
        _ = array_type(np.zeros((4, 5)))


@pytest.mark.parametrize(
    "annotation,strict",
    [
        (tuple[int, ...], False),
        (tuple[int, int, int], True),
        (tuple[3, 3, 3], True),
        (tuple[Literal[3], Literal[3], Literal[3]], True),
    ],
)
def test_shape_tuple(annotation, strict):
    """Shape can just be a literal tuple, as with numpy arrays"""
    array_type = NDArray[annotation, Any]

    _ = array_type(np.zeros((3, 3, 3)))
    if strict:
        with pytest.raises(ShapeError):
            _ = array_type(np.zeros((6, 7)))
