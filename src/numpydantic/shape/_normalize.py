from typing import Any, Literal, get_args, get_origin

from numpydantic.shape._range import _RangeDim
from numpydantic.shape._string import (
    StrShapeExpression,
    _str_dim_to_tuple,
    normalize_string_shape_expression,
    validate_string_shape_expression,
)
from numpydantic.shape._variadic import Unpack, _is_any_shape


def normalize_shape_expression(expression: StrShapeExpression | tuple) -> tuple:
    """
    Normalize a shape expression to its canonical tuple form.

    Accepts:
    - Old nptyping style string expressions
    - Bare args (like `Shape[1, 2]`)
    - Valid tuple form (`Shape[Literal[1], Literal[2]]`)
    """
    # unpack singletons to handle string format - a single string or a `Literal["1, 2"]`
    if isinstance(expression, tuple) and len(expression) == 1:
        expression = expression[0]
    if (
        get_origin(expression) is Literal
        and len(args := get_args(expression)) == 1
        and isinstance(args[0], str)
    ):
        expression = args[0]

    if isinstance(expression, str):
        return _normalize_tuple(normalize_string_shape_expression(expression))
    else:
        if not isinstance(expression, tuple):
            expression = (expression,)
        return _normalize_tuple(expression)


def validate_shape_expression(expression: StrShapeExpression | tuple) -> tuple:
    if isinstance(expression, str):
        return validate_string_shape_expression(expression)
    else:
        raise NotImplementedError()


def _normalize_tuple(expression: tuple) -> tuple:
    """
    Normalize tuple args to a nptyping string.

    A trailing ``...`` repeats the dimension before it,
    which becomes a ``"{dim}..."`` term meaning "zero or more of {dim}".
    The string ``"..."`` means any number of dimensions of any size.

    FIXME: This whole 'convert everything down to a string' thing has gotta go
    """

    splat_last = False
    if expression[-1] is ... and len(expression) > 2:
        splat_last = True
        expression = expression[:-1]
    normalized = tuple(_normalize_tuple_item(item) for item in expression)

    if splat_last:
        normalized = (*normalized, *tuple[normalized[-1], ...])

    if _is_any_shape(expression):
        return Any
    return normalized


def _normalize_tuple_item(item: str | int | type | Any) -> type:
    if get_origin(item) is Literal:
        item = get_args(item)

    if isinstance(item, str):
        if "," in item:
            return tuple(_str_dim_to_tuple(i) for i in item.split(","))
        else:
            return _str_dim_to_tuple(item)
    elif isinstance(item, int):
        return Literal[item]
    elif isinstance(item, _RangeDim):
        return item.annotation
    elif get_origin(item) is Unpack:
        return next(iter(get_args(item)[0]))
    elif isinstance(item, tuple):
        if len(item) == 1:
            return _normalize_tuple_item(item[0])
        else:
            return tuple(_normalize_tuple_item(item) for item in item)
    else:
        return item
