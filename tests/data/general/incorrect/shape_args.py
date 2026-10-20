import numpy as np

from numpydantic import NDArray, Shape


def f(frame: NDArray[Shape[1080, 1920], np.uint8]) -> int: ...


rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
f(rgb)
