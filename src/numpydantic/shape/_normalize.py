import sys
from typing import Any, Literal, get_args, get_origin

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._range import _RangeDim
from numpydantic.shape._string import (
    StrShapeExpression,
    _str_dim_to_tuple,
    normalize_string_shape_expression,
    validate_string_shape_expression,
)
from numpydantic.shape._variadic import _is_any_shape, _variadic_item

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack


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
    """
    expression = _flatten(expression)
    normalized = tuple(_normalize_tuple_item(item) for item in expression)
    _check_positions(normalized)
    if normalized[-1] is Ellipsis and len(normalized) > 2:
        normalized = (*normalized[:-1], *tuple[normalized[-2], ...])
    if _is_any_shape(normalized):
        return Any
    return normalized


def _flatten(expression: tuple) -> tuple:
    """Unpacked fixed-length tuples are just their items: tuple[A, *tuple[B, C]]"""
    flat = []
    for item in expression:
        # typing's own attribute for the items of an unpacked tuple, as * or Unpack
        args = getattr(item, "__typing_unpacked_tuple_args__", None)
        if args is not None and not (args and args[-1] is Ellipsis):
            flat.extend(_flatten(args))
        else:
            flat.append(item)
    return tuple(flat)


def _check_positions(expression: tuple) -> None:
    """``...`` and *tuple[X, ...] can only be the last item in a shape"""
    *init, last = expression
    if any(item is Ellipsis or _variadic_item(item) is not None for item in init):
        raise InvalidShapeExpressionError(
            "... and unpacked dimensions like *AnyDims must be the last item in a shape"
        )
    if last is Ellipsis and not init:
        raise InvalidShapeExpressionError("... must follow a dimension")


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
