import types
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._range import _RangeDim
from numpydantic.shape._variadic import _split_variadic


def check_typevar(var: TypeVar) -> None:
    """
    Raise if a shape variable's bound or constraints aren't sizes:
    ``int``, a ``Literal`` int, a :class:`.Range`, or a union of those
    """
    if var.__bound__ is None and not var.__constraints__:
        raise InvalidShapeExpressionError(
            "Shape variables must be bound or constrained to sizes, "
            f"like TypeVar({var.__name__!r}, bound=int), got {var!r}"
        )
    for spec in typevar_specs(var):
        if not _is_size(spec):
            raise InvalidShapeExpressionError(
                f"Shape variable {var!r} can only be bound or constrained to sizes, "
                f"like int, a Literal int, or a Range, got {spec!r}"
            )


def _is_size(spec: Any) -> bool:
    if spec is int:
        return True
    origin = get_origin(spec)
    if origin is Literal:
        return all(type(value) is int for value in get_args(spec))
    # a range, like Annotated[int, Interval(ge=2)]
    if origin is Annotated:
        return _is_size(get_args(spec)[0])
    return False


def typevar_specs(var: TypeVar) -> tuple[Any, ...]:
    """
    The specs a shape variable's size can fit: its constraints, or its bound.
    Unions are split into their members, since a size only has to fit one of them.
    """
    specs = var.__constraints__ or (var.__bound__,)
    return tuple(member for spec in specs for member in _members(spec))


def _members(spec: Any) -> tuple[Any, ...]:
    if isinstance(spec, _RangeDim):
        return (spec.annotation,)
    if get_origin(spec) in (Union, types.UnionType):
        return tuple(member for arg in get_args(spec) for member in _members(arg))
    return (spec,)


def typevar_positions(target: Any) -> dict[int, str]:
    """
    The name of the shape variable at each index of a normalized shape.

    A repeated shape variable, as in ``Shape[N, ...]``, is at the first index it
    repeats from, which a shape that matches has all the same size from.
    """
    if target is Any:
        return {}
    fixed, tail = _split_variadic(get_args(target))
    dims = fixed if tail is None else (*fixed, tail)
    positions = {}
    for index, dim in enumerate(dims):
        if get_origin(dim) is Annotated:
            dim = get_args(dim)[0]
        if isinstance(dim, TypeVar):
            positions[index] = dim.__name__
    return positions
