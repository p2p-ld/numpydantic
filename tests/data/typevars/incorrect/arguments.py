"""Shape typevars given different sizes, through each kind of argument"""

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
    x4: NDArray[Shape[L[4]], F],
    three: L[3],
    four: L[4],
    list3: list[NDArray[Shape[L[3]], F]],
    dict3: dict[str, NDArray[Shape[L[3]], F]],
) -> None:
    # an optional array with a different size from the other arguments fails
    maybe(x3, x4, x4)  # E[mypy]: Shape typevar "N" is 3 in "a" but 4 in "b"
    # None for an optional array doesn't stop the other arguments being checked
    maybe(None, x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "b" but 4 in "c"
    # a dict value with a different size from another argument fails
    named({"a": x3}, x4)  # E[mypy]: Shape typevar "N" is 3 in "arrays" but 4 in "other"
    # dict values with different sizes fail
    named({"a": x3, "b": x4}, x3)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # an unpacked dict's values with a different size from another value fail
    named({**dict3, "b": x4}, x3)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # list items with a different size from another argument fail
    many([x3, x3], x4)  # E[mypy]: Shape typevar "N" is 3 in "arrays" but 4 in "other"
    # list items with different sizes fail
    many([x3, x4], x3)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # an unpacked list's items with a different size from another item fail
    many([*list3, x4], x3)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # a list variable with a different size from another argument fails.
    # Lists are invariant, so pyright refuses this too
    many(list3, x4)  # E: Shape typevar "N" is 3 in "arrays" but 4 in "other"
    # nested list items with different sizes fail, and pyright refuses this too
    nested([[x3], [x4]])  # E: Shape typevar "N" is both 3 and 4 in "arrays"
    # tuple items with different sizes fail
    pair((x3, x4))  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # *args with different sizes fail
    stack(x3, x4)  # E[mypy]: Shape typevar "N" is both 3 and 4 in "arrays"
    # a plain parameter and an array that give the typevar different sizes fail
    sized(three, x4)  # E[mypy]: Shape typevar "N" is 3 in "n" but 4 in "a"
    # a size typevar is still checked next to a typevar that isn't a size
    mixed(three, four, x3, x4)  # E[mypy]: Shape typevar "N" is 3 in "x" but 4 in "y"
