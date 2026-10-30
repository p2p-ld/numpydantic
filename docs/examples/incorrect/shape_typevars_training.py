from typing import Literal as L
from typing import TypeVar

import numpy as np

from numpydantic import NDArray, Shape

Samples = TypeVar("Samples", bound=int)
Features = TypeVar("Features", bound=int)


def fit_predict(
    X: NDArray[Shape[Samples, Features], np.float64],
    y: NDArray[Shape[Samples], np.float64],
) -> NDArray[Shape[Samples], np.float64]:
    # a real model would be fit here
    return y


def train(
    X: NDArray[Shape[L[100], L[4]], np.float64],
    y: NDArray[Shape[L[90]], np.float64],
) -> None:
    fit_predict(X, y)
