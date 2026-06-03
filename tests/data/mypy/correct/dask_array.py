from typing import reveal_type

import dask.array as da

from numpydantic import NDArray, Shape


def array() -> NDArray[Shape[1, 2, 3]]:
    return da.zeros((1, 2, 3))


x: NDArray[Shape[1, 2, 3]] = array()
y: NDArray[Shape[1, 2, 3]] = da.zeros((1, 2, 3))
reveal_type(array)
reveal_type(x)
reveal_type(y)
