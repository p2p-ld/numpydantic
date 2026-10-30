"""
transposed reverses shapes with up to 5 dimensions.

These rely on numpy typing .T as keeping the shape it was given.
"""

from typing import Annotated as A
from typing import Any, assert_type
from typing import Literal as L

import numpy as np

from numpydantic import NDArray, Repeat, Shape, transposed

f = np.float64


def reverses(
    d0: NDArray[Shape[()], f],
    d1: NDArray[Shape[L[1]], f],
    d2: NDArray[Shape[L[1], L[2]], f],
    d3: NDArray[Shape[L[1], L[2], L[3]], f],
    d4: NDArray[Shape[L[1], L[2], L[3], L[4]], f],
    d5: NDArray[Shape[L[1], L[2], L[3], L[4], L[5]], f],
    ints: NDArray[Shape[int, L[2]], f],
    labelled: NDArray[Shape[A[L[3], "x"], A[L[4], "y"]], f],
) -> None:
    assert_type(transposed(d0.T), NDArray[Shape[()], f])
    assert_type(transposed(d1.T), NDArray[Shape[L[1]], f])
    assert_type(transposed(d2.T), NDArray[Shape[L[2], L[1]], f])
    assert_type(transposed(d3.transpose()), NDArray[Shape[L[3], L[2], L[1]], f])
    assert_type(transposed(d4.T), NDArray[Shape[L[4], L[3], L[2], L[1]], f])
    assert_type(transposed(d5.T), NDArray[Shape[L[5], L[4], L[3], L[2], L[1]], f])
    assert_type(transposed(ints.T), NDArray[Shape[L[2], int], f])
    assert_type(transposed(labelled.T), NDArray[Shape[L[4], L[3]], f])


def unknown(
    d6: NDArray[Shape[L[1], L[2], L[3], L[4], L[5], L[6]], f],
    any_ndim: NDArray[Shape[int, ...], f],
    any_shape: NDArray[Any, f],
) -> None:
    assert_type(transposed(d6.T), NDArray[Shape[Any, ...], f])
    assert_type(transposed(any_shape.T), NDArray[Shape[Any, ...], f])
    # pyright gives tuple[int, ...] and mypy gives tuple[Any, ...]
    reversed_ndim: NDArray[Shape[*Repeat], f] = transposed(any_ndim.T)
    print(reversed_ndim)


def assigns(a: NDArray[Shape[L[3], L[4]], np.uint8]) -> None:
    b: NDArray[Shape[L[4], L[3]], np.uint8] = transposed(a.T)
    print(b)


def any_dtype(
    omitted: NDArray[Shape[L[3], L[4]]],
    explicit: NDArray[Shape[L[3], L[4]], Any],
    raw_any: np.ndarray[tuple[L[3], L[4]], Any],
    raw_dtype_any: np.ndarray[tuple[L[3], L[4]], np.dtype[Any]],
) -> None:
    assert_type(transposed(omitted.T), NDArray[Shape[L[4], L[3]]])
    assert_type(transposed(explicit.T), NDArray[Shape[L[4], L[3]], Any])
    assert_type(transposed(raw_any.T), np.ndarray[tuple[L[4], L[3]], Any])
    assert_type(transposed(raw_dtype_any.T), NDArray[Shape[L[4], L[3]], Any])
