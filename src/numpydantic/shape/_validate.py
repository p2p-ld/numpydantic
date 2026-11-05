import types
from collections.abc import Mapping
from functools import lru_cache
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from annotated_types import Interval

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._typevars import typevar_specs
from numpydantic.shape._variadic import _split_variadic


def validate_shape(
    shape: tuple[int, ...],
    target: Any,
    typevar_sizes: Mapping[str, int] | None = None,
) -> bool:
    """
    Check whether the given shape corresponds to the given shape_expression.
    :param shape: the shape in question.
    :param target: the shape expression to which shape is tested.
    :param typevar_sizes: sizes that shape typevars already have, by name
    :return: True if the given shape corresponds to shape_expression.
    """
    if typevar_sizes:
        return _validate_shape(shape, target, typevar_sizes)
    return _validate_shape_cached(shape, target)


@lru_cache
def _validate_shape_cached(shape: tuple[int, ...], target: Any) -> bool:
    return _validate_shape(shape, target, {})


def _validate_shape(
    shape: tuple[int, ...], target: Any, typevar_sizes: Mapping[str, int]
) -> bool:
    if target is Any:
        return True

    fixed, tail = _split_variadic(get_args(target))
    if len(shape) < len(fixed) or (tail is None and len(shape) != len(fixed)):
        return False
    dims = [*fixed, *[tail] * (len(shape) - len(fixed))]
    # A copy, so sizes seen in this shape, as in Shape[N, N],
    # aren't kept if it doesn't match
    typevar_sizes = dict(typevar_sizes)
    return all(
        _validate_dim(size, dim, typevar_sizes) for size, dim in zip(shape, dims)
    )


def _validate_dim(
    size: int, dim: Any, typevar_sizes: dict[str, int] | None = None
) -> bool:
    if dim is int or dim is Any:
        return True
    if isinstance(dim, TypeVar):
        typevar_sizes = {} if typevar_sizes is None else typevar_sizes
        if dim.__name__ in typevar_sizes:
            return size == typevar_sizes[dim.__name__]
        if not any(_validate_dim(size, fits) for fits in typevar_specs(dim)):
            return False
        typevar_sizes[dim.__name__] = size
        return True
    origin = get_origin(dim)
    if origin is Literal:
        return size in get_args(dim)
    if origin in (Union, types.UnionType):
        return any(
            _validate_dim(size, member, typevar_sizes) for member in get_args(dim)
        )
    if origin is Annotated:
        base, *metadata = get_args(dim)
        return _validate_dim(size, base, typevar_sizes) and all(
            _validate_metadata(size, meta) for meta in metadata
        )
    raise InvalidShapeExpressionError(f"Unknown shape dimension {dim!r}")


def _validate_metadata(size: int, meta: Any) -> bool:
    if isinstance(meta, Interval):
        return (
            (meta.ge is None or size >= meta.ge)
            and (meta.gt is None or size > meta.gt)
            and (meta.le is None or size <= meta.le)
            and (meta.lt is None or size < meta.lt)
        )
    # labels and other metadata don't constrain the size
    return True
