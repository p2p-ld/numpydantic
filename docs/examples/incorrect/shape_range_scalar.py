import numpy as np

from numpydantic import NDArray, Shape


def make_array() -> NDArray[Shape[1, 2, 3], np.uint8]:
    return np.ones((1, 2, 3), dtype=np.uint8)


x: NDArray[Shape["1-3, 1-4, 10-20"], np.uint8] = make_array()
