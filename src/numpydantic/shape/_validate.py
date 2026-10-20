from functools import lru_cache
from typing import Annotated, Any, Literal, get_args, get_origin

from annotated_types import Interval

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._variadic import _split_variadic


@lru_cache
def validate_shape(shape: tuple[int, ...], target: Any) -> bool:
    """
    Check whether the given shape corresponds to the given shape_expression.
    :param shape: the shape in question.
    :param target: the shape expression to which shape is tested.
    :return: True if the given shape corresponds to shape_expression.
    """
    if target is Any:
        return True

    fixed, tail = _split_variadic(get_args(target))
    if len(shape) < len(fixed) or (tail is None and len(shape) != len(fixed)):
        return False
    if not all(_validate_dim(size, dim) for size, dim in zip(shape, fixed)):
        return False
    if tail is None:
        return True
    return all(_validate_dim(size, tail) for size in shape[len(fixed) :])


def _validate_dim(size: int, dim: Any) -> bool:
    if dim is int:
        return True
    origin = get_origin(dim)
    if origin is Literal:
        return size in get_args(dim)
    if origin is Annotated:
        base, *metadata = get_args(dim)
        return _validate_dim(size, base) and all(
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
