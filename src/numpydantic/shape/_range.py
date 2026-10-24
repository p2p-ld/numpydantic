from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from annotated_types import Interval

from numpydantic.exceptions import InvalidShapeExpressionError


class Range:
    """
    An inclusive range of sizes for one dimension.

    Bounds are ``Literal`` ints, and ``int`` leaves that side open::

        Shape[Range[L[2], L[4]], Range[L[2], int], Range[int, L[4]]]

    To type checkers, a range is just an ``int``.
    """

    def __class_getitem__(cls, item: Any) -> "_RangeDim":
        if not isinstance(item, tuple) or len(item) != 2:
            raise InvalidShapeExpressionError(
                f"Range takes a low and a high bound, got {item!r}"
            )
        low, high = (_range_bound(bound) for bound in item)
        if low is not None and high is not None and low > high:
            raise InvalidShapeExpressionError(
                f"Range low bound {low} is greater than high bound {high}"
            )
        return _RangeDim(low, high)


@dataclass(frozen=True)
class _RangeDim:
    """A parameterized Range, which is Annotated[int, Interval(...)] in a shape"""

    low: int | None = None
    high: int | None = None

    def __repr__(self) -> str:
        low = "int" if self.low is None else f"L[{self.low}]"
        high = "int" if self.high is None else f"L[{self.high}]"
        return f"Range[{low}, {high}]"

    @property
    def annotation(self) -> type:
        return Annotated[int, Interval(ge=self.low, le=self.high)]


def _range_bound(bound: Any) -> int | None:
    if bound is int:
        return None
    if get_origin(bound) is Literal and len(get_args(bound)) == 1:
        value = get_args(bound)[0]
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise InvalidShapeExpressionError(
        "Range bounds must be a non-negative Literal int like L[2], "
        f"or int for no bound, got {bound!r}"
    )
