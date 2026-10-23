from typing import Any, Literal, Unpack, get_args, get_origin

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._range import _RangeDim
from numpydantic.shape._string import (
    _str_dim_to_tuple,
    normalize_string_shape_expression,
)
from numpydantic.shape._variadic import _is_any_shape, _variadic_item


def normalize_shape_expression(expression: Any) -> tuple | Any:
    """
    Normalize the arguments of ``Shape[...]`` to the arguments of the equivalent
    ``tuple[...]``, or ``Any`` if any shape is allowed.

    Accepts:

    - Type expressions, like ``Shape[L[1], int, Range[L[2], L[4]], *AnyDims]``
    - Bare ints, like ``Shape[1, 2]``
    - Deprecated string expressions, like ``Shape["1, *, 2-4, ..."]``
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
        expression = normalize_string_shape_expression(expression)
    elif not isinstance(expression, tuple):
        expression = (expression,)
    return _normalize_tuple(expression)


def _normalize_tuple(expression: tuple) -> tuple | Any:
    expression = _flatten(expression)
    if not expression:
        # a 0-d array, as with tuple[()]
        return ()
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


def _normalize_tuple_item(item: Any) -> Any:
    if get_origin(item) is Literal:
        args = get_args(item)
        if len(args) > 1:
            raise InvalidShapeExpressionError(
                f"A dimension can only have one exact size, got {item!r}"
            )
        if isinstance(args[0], int):
            return item
        # TODO: just support multivalued literals as unions
        item = args[0]

    if isinstance(item, str):
        return _str_dim_to_tuple(item)
    elif isinstance(item, int):
        return Literal[item]
    elif isinstance(item, _RangeDim):
        return item.annotation
    elif get_origin(item) is Unpack:
        # Unpack[X] is *X
        return next(iter(get_args(item)[0]))
    return item
