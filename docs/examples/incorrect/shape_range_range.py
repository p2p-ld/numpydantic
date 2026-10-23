import numpy as np

from numpydantic import NDArray, Shape


def make_range() -> NDArray[Shape["3-5, 3-5"]]:
    return np.ones((4, 4))


x: NDArray[Shape["6-8, 6-8"]] = make_range()
