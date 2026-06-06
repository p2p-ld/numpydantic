import numpy as np

from numpydantic import NDArray, Shape

z = np.zeros((0, 1), dtype=np.uint8)


def mask_tuple(shape: tuple[int, ...]) -> NDArray[Shape["* x, * y"], np.uint8]:
    return np.zeros((shape[0], shape[1]), np.uint8)


def mask_ndarray(shape: np.ndarray) -> NDArray[Shape["* x, * y"], np.uint8]:
    return np.zeros(shape[0, 1], np.uint8)


x: NDArray[Shape["*, *"], np.uint8] = mask_tuple((1, 2))
y: NDArray[Shape["*, *"], np.uint8] = mask_ndarray(np.array([1, 2]))
