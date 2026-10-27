"""unliteral keeps the number of dimensions of shapes with up to 5 dimensions"""

from typing import Annotated as A
from typing import Any, assert_type
from typing import Literal as L

import numpy as np

from numpydantic import AnyDims, NDArray, Range, Shape, unliteral

f = np.float64


def widens(
    d0: NDArray[Shape[()], f],
    d1: NDArray[Shape[L[1]], f],
    d2: NDArray[Shape[L[1], L[2]], f],
    d3: NDArray[Shape[L[1], L[2], L[3]], f],
    d4: NDArray[Shape[L[1], L[2], L[3], L[4]], f],
    d5: NDArray[Shape[L[1], L[2], L[3], L[4], L[5]], f],
    ints: NDArray[Shape[int, L[2]], f],
    ranged: NDArray[Shape[Range[L[2], L[4]], L[3]], f],
    labelled: NDArray[Shape[A[L[3], "x"], A[L[4], "y"]], f],
) -> None:
    assert_type(unliteral(d0), NDArray[Shape[()], f])
    assert_type(unliteral(d1), NDArray[Shape[int], f])
    assert_type(unliteral(d2), NDArray[Shape[int, int], f])
    assert_type(unliteral(d3), NDArray[Shape[int, int, int], f])
    assert_type(unliteral(d4), NDArray[Shape[int, int, int, int], f])
    assert_type(unliteral(d5), NDArray[Shape[int, int, int, int, int], f])
    assert_type(unliteral(ints), NDArray[Shape[int, int], f])
    assert_type(unliteral(ranged), NDArray[Shape[int, int], f])
    assert_type(unliteral(labelled), NDArray[Shape[int, int], f])


def unknown(
    d6: NDArray[Shape[L[1], L[2], L[3], L[4], L[5], L[6]], f],
    any_ndim: NDArray[Shape[int, ...], f],
    mixed: NDArray[Shape[L[3], *AnyDims], f],
    any_shape: NDArray[Any, f],
) -> None:
    assert_type(unliteral(d6), NDArray[Shape[Any, ...], f])
    assert_type(unliteral(mixed), np.ndarray[tuple[int, ...], np.dtype[f]])
    assert_type(unliteral(any_shape), NDArray[Shape[Any, ...], f])
    # pyright gives tuple[int, ...] and mypy gives tuple[Any, ...]
    widened_ndim: NDArray[Shape[*AnyDims], f] = unliteral(any_ndim)
    print(widened_ndim)


def any_dtype(
    omitted: NDArray[Shape[L[3], L[4]]],
    explicit: NDArray[Shape[L[3], L[4]], Any],
    raw_any: np.ndarray[tuple[L[3], L[4]], Any],
    raw_dtype_any: np.ndarray[tuple[L[3], L[4]], np.dtype[Any]],
) -> None:
    assert_type(unliteral(omitted), NDArray[Shape[int, int]])
    assert_type(unliteral(explicit), NDArray[Shape[int, int], Any])
    assert_type(unliteral(raw_any), np.ndarray[tuple[int, int], Any])
    assert_type(unliteral(raw_dtype_any), NDArray[Shape[int, int], Any])
