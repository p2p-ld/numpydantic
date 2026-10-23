"""
The shape of an array, written like a tuple type, one item per dimension:

- ``L[3]`` (a ``typing.Literal``) - a dimension of exactly length 3
- ``int`` - a dimension of any length
- ``Range[L[2], L[4]]`` - a dimension of length 2 to 4, inclusive.
  ``int`` leaves that side open, e.g. ``Range[L[2], int]`` is at least 2.
- ``*AnyDims`` - any number of dimensions of any length, as the last item.
- ``...`` - any number of dimensions matching the one before it, as the last item.
  ``Shape[L[2], ...]`` is any number of 2-length dimensions, as with
  ``tuple[L[2], ...]``, and ``Shape[L[1], L[2], ...]`` is a 1-length dimension,
  a 2-length dimension, and any number of additional 2-length dimensions.

To type checkers, ``Shape`` is ``tuple``, so ``Shape[...]`` is a valid type expression.
At runtime, ``Shape[...]`` normalizes its arguments to the equivalent ``tuple[...]``.
"""

from typing import Any, TypeAlias

from numpydantic.shape._normalize import normalize_shape_expression
from numpydantic.shape._string import _warn_string_expression

AnyDims: TypeAlias = tuple[int, ...]
"""
Any number of dimensions of any size, including none.

Use it unpacked as the last item in a shape::

    Shape[L[3], *AnyDims]

or with ``Unpack`` before python 3.11::

    Shape[L[3], Unpack[AnyDims]]
"""


class Shape:
    """
    The shape of an array, as a tuple of dimensions.

    ``Shape[...]`` returns the equivalent ``tuple[...]``,
    or ``Any`` when it allows any shape:

    >>> from typing import Literal as L
    >>> Shape[L[2], int]
    tuple[typing.Literal[2], int]
    >>> Shape[L[2], int] == tuple[L[2], int]
    True
    """

    def __class_getitem__(cls, item: Any) -> type[tuple] | Any:
        _warn_string_expression(item)
        normalized = normalize_shape_expression(item)
        if normalized is Any:
            return Any
        return tuple[normalized]

    def __new__(cls, *args: Any) ->  type[tuple] | Any:
        """``Shape(3, 4)`` is ``Shape[3, 4]``, kept for backwards compatibility"""
        return cls[args]
