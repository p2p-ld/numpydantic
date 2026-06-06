import numpy as np

from numpydantic import NDArray, Shape


def make_array() -> NDArray[Shape[2, 3, 4], np.uint8]:
    return np.ones((1, 2, 3), dtype=np.uint8)


x: NDArray[Shape[5, 6, 7], np.uint8] = make_array()
