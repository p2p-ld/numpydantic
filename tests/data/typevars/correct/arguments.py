"""Shape typevars given sizes that agree, through each kind of argument"""

from collections.abc import Mapping
from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Shape

N = TypeVar("N", bound=int)
T = TypeVar("T")
F = np.float64


def maybe(
    a: NDArray[Shape[N], F] | None, b: NDArray[Shape[N], F], c: NDArray[Shape[N], F]
) -> None: ...
def named(
    arrays: Mapping[str, NDArray[Shape[N], F]], other: NDArray[Shape[N], F]
) -> None: ...
def many(arrays: list[NDArray[Shape[N], F]], other: NDArray[Shape[N], F]) -> None: ...
def nested(arrays: list[list[NDArray[Shape[N], F]]]) -> None: ...
def pair(arrays: tuple[NDArray[Shape[N], F], NDArray[Shape[N], F]]) -> None: ...
def stack(*arrays: NDArray[Shape[N], F]) -> None: ...
def sized(n: N, a: NDArray[Shape[N], F]) -> None: ...
def mixed(
    a: T, b: T, x: NDArray[Shape[N], F], y: NDArray[Shape[N], F]
) -> None: ...


def use(
    x3: NDArray[Shape[L[3]], F],
    y3: NDArray[Shape[L[3]], F],
    three: L[3],
    four: L[4],
    list3: list[NDArray[Shape[L[3]], F]],
    dict3: dict[str, NDArray[Shape[L[3]], F]],
) -> None:
    # an optional array with the same size as the other arguments passes
    maybe(x3, y3, y3)
    # None for an optional array gives the typevar no size, and the other arguments agree
    maybe(None, x3, y3)
    # dict values with the same size as each other and another argument pass
    named({"a": x3, "b": y3}, y3)
    # an unpacked dict's values with the same size as the other values pass
    named({**dict3, "b": y3}, y3)
    # list items with the same size as each other and another argument pass
    many([x3, y3], y3)
    # an unpacked list's items with the same size as the other items pass
    many([*list3, y3], y3)
    # a list variable with the same size as another argument passes
    many(list3, y3)
    # nested list items with the same size pass
    nested([[x3], [y3]])
    # tuple items with the same size pass
    pair((x3, y3))
    # *args with the same size pass
    stack(x3, y3)
    # a plain parameter typed with the typevar matches an array of that size
    sized(three, x3)
    # a typevar that isn't a size can be given different literals, while the size agrees
    mixed(three, four, x3, y3)
