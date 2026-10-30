"""Each kind of bound and constraint a shape typevar can have"""

from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Range, Shape

Int = TypeVar("Int", bound=int)
Literal3 = TypeVar("Literal3", bound=L[3])
Literals = TypeVar("Literals", bound=L[2, 3])
Union23 = TypeVar("Union23", bound=L[2] | L[3])
Ranged = TypeVar("Ranged", bound=Range[L[2], L[4]])
Constrained = TypeVar("Constrained", L[3], L[4])
IntOr3 = TypeVar("IntOr3", int, L[3])
Ranges = TypeVar("Ranges", Range[L[1], L[2]], Range[L[5], L[6]])
F = np.float64


# a typevar bound to int can be used in a shape
def int_bound(a: NDArray[Shape[Int], F]) -> None: ...
# a typevar bound to a literal size can be used in a shape
def literal_bound(a: NDArray[Shape[Literal3], F]) -> None: ...
# a typevar bound to a multi-value literal can be used in a shape
def literals_bound(a: NDArray[Shape[Literals], F]) -> None: ...
# a typevar bound to a union of literals can be used in a shape
def union_bound(a: NDArray[Shape[Union23], F]) -> None: ...
# a typevar bound to a range can be used in a shape
def range_bound(a: NDArray[Shape[Ranged], F]) -> None: ...
# a typevar constrained to literal sizes can be used in a shape
def literal_constraints(a: NDArray[Shape[Constrained], F]) -> None: ...
# a typevar constrained to int and a literal size can be used in a shape
def int_constraint(a: NDArray[Shape[IntOr3], F]) -> None: ...
# a typevar constrained to ranges can be used in a shape
def range_constraints(a: NDArray[Shape[Ranges], F]) -> None: ...
