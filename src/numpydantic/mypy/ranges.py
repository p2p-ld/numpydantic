"""
Shape ranges in the mypy plugin.

mypy can't be extended with new types, so a range is stored as a :class:`.RangeValue`
inside a ``Literal`` type, and checked when mypy compares literals.

mypy doesn't keep that value when it stores types in its cache,
so modules with ranges are found with :func:`.module_has_ranges`
and rechecked on every run instead.
"""

from typing import Any

from annotated_types import Interval
from mypy.nodes import (
    Decorator,
    FuncBase,
    MypyFile,
    SymbolTable,
    TypeAlias,
    TypeInfo,
    Var,
)
from mypy.type_visitor import TypeQuery
from mypy.types import LiteralType


class RangeValue(str):
    """
    An unholy python type used to overcome the inability to declare custom mypy types
    for checking ranges, otherwise we would have to do something like
    Union[Literal[1], Literal[2], Literal[3], ...]

    It is used as the value of a ``LiteralType``.
    mypy checks whether one literal is a subtype of another with
    ``subtype.value == supertype.value``, so ``__eq__`` does the range check instead.
    It's a ``str`` so mypy can store it in its cache.

    See: https://github.com/python/mypy/issues/16497#issuecomment-4570099557
    """

    low: int | None
    high: int | None

    def __new__(cls, interval: Interval):  # noqa: D102
        low = interval.ge if interval.gt is None else interval.gt + 1
        high = interval.le if interval.lt is None else interval.lt - 1
        inst = super().__new__(
            cls, f"{'*' if low is None else low}-{'*' if high is None else high}"
        )
        inst.low = low
        inst.high = high
        return inst

    def __hash__(self):
        # Equal to the ints within the range, but can't hash the same as all of them
        return hash((self.low, self.high))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RangeValue):
            # both sides are ranges, so self is on the left, the candidate subtype,
            # and has to fit within other
            return other._contains(self.low, self.high)
        elif isinstance(other, int) and not isinstance(other, bool):
            # can't tell which side of the comparison self is on,
            # so any size within the range is equal to it
            return self._contains(other, other)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        # str has its own __ne__, which would otherwise compare as strings
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def _contains(self, low: int | None, high: int | None) -> bool:
        """Whether the range from low to high (None for no bound) is within this one"""
        low_fits = self.low is None or (low is not None and low >= self.low)
        high_fits = self.high is None or (high is not None and high <= self.high)
        return low_fits and high_fits


def module_has_ranges(module: MypyFile) -> bool:
    """Whether anything defined in a module has a type with a range in it"""
    return _names_have_ranges(module.names, module.fullname)


class _HasRange(TypeQuery[bool]):
    """
    Whether a type contains a range, like ndarray[tuple[Literal['2-4']], ...]

    (Not BoolTypeQuery, which mypyc doesn't allow python classes to subclass)
    """

    def strategy(self, items: list[bool]) -> bool:
        return any(items)

    def visit_literal_type(self, t: LiteralType, /) -> bool:
        return isinstance(t.value, RangeValue)


def _names_have_ranges(names: SymbolTable, module_name: str) -> bool:
    for sym in names.values():
        node = sym.node
        # imported names are cached as references to the module that defines them
        if node is None or not node.fullname.startswith(f"{module_name}."):
            continue
        if isinstance(node, TypeInfo):
            if _names_have_ranges(node.names, module_name):
                return True
            types = node.bases
        elif isinstance(node, Decorator):
            types = [node.var.type, node.func.type]
        elif isinstance(node, (Var, FuncBase)):
            types = [node.type]
        elif isinstance(node, TypeAlias):
            types = [node.target]
        else:
            continue
        # the query keeps state, so each type gets a new one
        if any(t is not None and t.accept(_HasRange()) for t in types):
            return True
    return False
