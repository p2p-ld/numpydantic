"""Shape typevars have to be bound or constrained to sizes"""

from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Shape

Unbounded = TypeVar("Unbounded")
Float = TypeVar("Float", bound=float)
Bool = TypeVar("Bool", bound=L[True])
UnionStr = TypeVar("UnionStr", bound=L[2] | str)
Str = TypeVar("Str", str, L[3])
F = np.float64


# a typevar without a bound or constraints can't be used in a shape
def unbounded(a: NDArray[Shape[Unbounded], F]) -> None: ...  # E[mypy]: Shape typevars must be bound or constrained to sizes, like TypeVar('Unbounded', bound=int), got ~Unbounded
# a typevar bound to a type that isn't a size can't be used in a shape
def float_bound(a: NDArray[Shape[Float], F]) -> None: ...  # E[mypy]: Shape typevar ~Float can only be bound or constrained to sizes, like int, a Literal int, or a Range, got float
# a typevar bound to a bool literal can't be used in a shape, since bools aren't sizes
def bool_bound(a: NDArray[Shape[Bool], F]) -> None: ...  # E[mypy]: Shape typevar ~Bool can only be bound or constrained to sizes, like int, a Literal int, or a Range, got typing.Literal[True]
# a typevar bound to a union with a member that isn't a size can't be used in a shape
def union_bound(a: NDArray[Shape[UnionStr], F]) -> None: ...  # E[mypy]: Shape typevar ~UnionStr can only be bound or constrained to sizes, like int, a Literal int, or a Range, got str
# a typevar with a constraint that isn't a size can't be used in a shape
def str_constraint(a: NDArray[Shape[Str], F]) -> None: ...  # E[mypy]: Shape typevar ~Str can only be bound or constrained to sizes, like int, a Literal int, or a Range, got str
