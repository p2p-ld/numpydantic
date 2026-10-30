from collections.abc import Iterator
from typing import Any, Unpack

from numpydantic.exceptions import InvalidShapeExpressionError


class _RepeatMeta(type):
    def __iter__(cls) -> Iterator[Any]:
        # unpacking bare Repeat, as in Shape[L[3], *Repeat]
        return iter((Unpack[tuple[int, ...]],))


class Repeat(metaclass=_RepeatMeta):
    """
    Shorthand for any number of dimensions, including none,
    as the last item in a shape.

    ``Repeat[X]`` is ``tuple[X, ...]``, and bare ``Repeat`` is ``tuple[int, ...]``,
    so these are the same as unpacking those tuples::

        Shape[L[3], *Repeat]          # 3, then any dimensions
        Shape[L[3], *Repeat[L[2]]]    # 3, then any number of 2s
        Shape[L[3], *Repeat[N]]       # 3, then any number of dimensions of size N
    """

    def __class_getitem__(cls, item: Any) -> Any:
        if isinstance(item, tuple):
            raise InvalidShapeExpressionError(
                f"Repeat takes one dimension, got {item!r}"
            )
        return tuple[item, ...]
