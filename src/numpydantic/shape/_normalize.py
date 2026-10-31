import types
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    get_args,
    get_origin,
)

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._range import _RangeDim
from numpydantic.shape._repeat import Repeat
from numpydantic.shape._string import (
    _str_dim_to_tuple,
    normalize_string_shape_expression,
)
from numpydantic.shape._typevars import check_typevar
from numpydantic.shape._variadic import (
    _is_any_shape,
    _unpacked_tuple_args,
    _variadic_item,
)


def normalize_shape_expression(expression: Any) -> tuple | Any:
    """
    Normalize the arguments of ``Shape[...]`` to the arguments of the equivalent
    ``tuple[...]``, or ``Any`` if any shape is allowed.

    Accepts:

    - Type expressions, like ``Shape[L[1], int, Range[L[2], L[4]], *Repeat]``
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
        normalized = (*normalized[:-1], _repeated(normalized[-2]))
    elif (
        len(normalized) == 1 and (repeated := _variadic_item(normalized[0])) is not None
    ):
        # tuple[*tuple[X, ...]] is tuple[X, ...]
        normalized = (repeated, ...)
    if _is_any_shape(normalized):
        return Any
    return normalized


def _flatten(expression: tuple) -> tuple:
    """Unpacked fixed-length tuples are just their items: tuple[A, *tuple[B, C]]"""
    flat = []
    for item in expression:
        args = _unpacked_tuple_args(item)
        if args is not None and not (args and args[-1] is Ellipsis):
            flat.extend(_flatten(args))
        else:
            flat.append(item)
    return tuple(flat)


def _check_positions(expression: tuple) -> None:
    """
    ``...`` and *tuple[X, ...] can only be the last item in a shape,
    and can only repeat a dimension
    """
    *init, last = expression
    if any(_repeats(item) for item in init):
        raise InvalidShapeExpressionError(
            "... and unpacked dimensions like *Repeat must be the last item in a shape"
        )
    if last is Ellipsis and not init:
        raise InvalidShapeExpressionError("... must follow a dimension")
    for item in expression:
        if get_origin(item) is Annotated:
            inner = get_args(item)[0]
        else:
            inner = _variadic_item(item)
        if _repeats(inner):
            raise InvalidShapeExpressionError(
                f"... and unpacked dimensions like *Repeat can't be nested: {item!r}"
            )


def _repeats(item: Any) -> bool:
    """Whether an item repeats a dimension, like ... or *tuple[X, ...]"""
    return item is Ellipsis or _variadic_item(item) is not None


def _repeated(dim: Any) -> Any:
    """
    A dimension repeated at the end of a shape, like ``*tuple[X, ...]``.

    It's written as ``Unpack[tuple[X, ...]]`` because when pydantic parametrizes
    a model, it keeps ``Unpack`` but turns ``*tuple[X, ...]`` into a plain tuple.
    """
    return Unpack[tuple[dim, ...]]


def _normalize_tuple_item(item: Any) -> Any:
    if get_origin(item) is Unpack:
        unpacked = get_args(item)[0]
        # TypeVarTuples aren't supported yet, so they're kept as they are
        if isinstance(unpacked, TypeVarTuple):
            return item
        # Unpack[X] is *X
        item = next(iter(unpacked))
    if get_origin(item) is Literal:
        args = get_args(item)
        if all(isinstance(arg, int) for arg in args):
            return item
        if len(args) > 1:
            raise InvalidShapeExpressionError(
                f"A dimension with several exact sizes can only have ints, got {item!r}"
            )
        # a string dimension, like L["2-4"]
        item = args[0]

    if isinstance(item, str):
        return _normalize_tuple_item(_str_dim_to_tuple(item))
    elif isinstance(item, int):
        return Literal[item]
    elif isinstance(item, _RangeDim):
        return item.annotation
    elif isinstance(item, TypeVar):
        check_typevar(item)
        return item
    elif item is int or item is Any or item is Ellipsis:
        return item
    elif (repeated := _variadic_item(item)) is not None:
        return _repeated(_normalize_tuple_item(repeated))
    elif get_origin(item) is Annotated:
        base, *metadata = get_args(item)
        return Annotated[(_normalize_tuple_item(base), *metadata)]
    elif get_origin(item) in (Union, types.UnionType):
        return Union[  # noqa: UP007
            tuple(_normalize_tuple_item(member) for member in get_args(item))
        ]
    elif item is Repeat or get_origin(item) is tuple:
        raise InvalidShapeExpressionError(
            f"Repeated dimensions have to be unpacked, like *Repeat, got {item!r}"
        )
    raise InvalidShapeExpressionError(f"Invalid shape dimension: {item!r}")
