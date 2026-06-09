import numpy as np

from numpydantic import NDArray, Shape

GRAYSCALE = NDArray[Shape["* x, * y"], np.uint8]
RGB = NDArray[Shape["* x, * y, 3 rgb"], np.uint8]


def read_rgb() -> RGB:
    return np.ones((1920, 1080, 3), dtype=np.uint8)


def read_grayscale() -> GRAYSCALE:
    return np.ones((1920, 1080), dtype=np.uint8)


def grayscale_mask(frame: GRAYSCALE) -> GRAYSCALE:
    # Probably something fancier than this...
    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    mask[frame > 5] = 1
    return mask


# this works
grayscale_mask(read_grayscale())

# this doesn't
grayscale_mask(read_rgb())
