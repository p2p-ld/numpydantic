"""
Declaration and validation functions for array shapes.

Mostly a mildly modified version of nptyping's
:func:`npytping.shape_expression.validate_shape`
and its internals to allow for extended syntax, including ranges of shapes.

Modifications from nptyping:

- **"..."** - In nptyping, ``'...'`` means "any number of dimensions with the same shape
  as the last dimension. ie ``Shape[2, ...]`` means "any number of 2-length
  dimensions. Here, ``'...'`` within a (deprecated) string expression means
  "any number of any-shape dimensions", which is now written ``*AnyDims``.
  Outside of a string, ``Shape[L[2], ...]`` means "any number of 2-length dimensions",
  as with ``tuple[L[2], ...]``, and ``Shape[L[1], L[2], ...]`` means
  "a 1-length dimension, a 2-length dimension, and any number of additional 2-length
  dimensions".
- **Ranges** - (inclusive) shape ranges are allowed. eg. to specify an array
  where the first dimension can be 2, 3, or 4 length::

     Shape["2-4, ..."]

  To specify a range with an unbounded min or max, use wildcards, eg. for
  an array with the first dimension at least length 2, and the second dimension
  at most length 5 (both inclusive)::

      Shape["2-*, *-5"]

"""

import sys
from typing import Any, Generic, TypeAlias

from numpydantic.shape._normalize import normalize_shape_expression
from numpydantic.shape._string import _warn_string_expression

if sys.version_info < (3, 11):
    from typing import Self

    from typing_extensions import TypeVarTuple, Unpack
else:
    from typing import Self, TypeVarTuple, Unpack

_TV = TypeVarTuple("_TV")

AnyDims: TypeAlias = tuple[int, ...]
"""
Any number of dimensions of any size, including none.

Use it unpacked as the last item in a shape:

    Shape[L[3], *AnyDims]

or with ``Unpack`` before python 3.11:

    Shape[L[3], Unpack[AnyDims]]
"""


class Shape(Generic[Unpack[_TV]]):
    """
    A container for shape expressions that describe the shape of an multi
    dimensional array.

    Simple example:

    >>> Shape['2, 2']
    Shape['2, 2']

    A Shape can be compared to a typing.Literal. You can use Literals in
    NDArray as well.

    >>> from typing import Literal

    >>> Shape['2, 2'] == Literal['2, 2']
    True

    A Shape can be constructed by calling for type checker compatibility

    >>> Shape['2, 2'] == Shape('2, 2')

    And its arguments can be passed as ``args``, with ints and strings as appropriate

    >>> Shape(2, 2, "...") == Shape("2, 2, ...")

    """

    def __class_getitem__(cls, item: Any) -> type[tuple]:
        _warn_string_expression(item)
        normalized = normalize_shape_expression(item)
        if normalized is Any:
            return Any
        else:
            return tuple[normalized]

    def __new__(cls, *args: Any) -> Self:
        return cls[args]
