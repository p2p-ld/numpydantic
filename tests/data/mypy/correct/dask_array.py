from typing import Literal as L

import dask.array as da

from numpydantic import NDArray, Shape


def array() -> NDArray[Shape[L[1], L[2], L[3]]]:
    return da.zeros((1, 2, 3))


x: NDArray[Shape[L[1], L[2], L[3]]] = array()
y: NDArray[Shape[L[1], L[2], L[3]]] = da.zeros((1, 2, 3))
