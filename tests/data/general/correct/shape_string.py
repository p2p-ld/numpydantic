import numpy as np

from numpydantic import NDArray, Shape

x: NDArray[Shape["1, 2, 3"]] = np.zeros((1, 2, 3))
