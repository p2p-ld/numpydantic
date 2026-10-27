from typing import TypeVar, TypeAlias

from dataclasses import dataclass

# Type checkers can't check ranges, so a range is just an int to them.
# The bounds only appear in the union so that the alias can be subscripted.
_Low = TypeVar("_Low", bound=int)
_High = TypeVar("_High", bound=int)
Range: TypeAlias = int | _Low | _High

@dataclass(frozen=True)
class _RangeDim:
    """A parameterized Range, which is Annotated[int, Interval(...)] in a shape"""

    low: int | None = None
    high: int | None = None

    def __repr__(self) -> str: ...
    @property
    def annotation(self) -> type: ...
