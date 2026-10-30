from collections.abc import Callable
from typing import Any
from typing import Literal as L

import numpy as np
import pytest
from pydantic import BaseModel, Field

from numpydantic import NDArray, Repeat, Shape
from numpydantic.dtype import Number


@pytest.fixture(scope="function")
def array_model() -> Callable[[tuple[int, ...], type | np.dtype], type[BaseModel]]:
    def _model(
        shape: tuple[int, ...] = (10, 10), dtype: type | np.dtype = float
    ) -> type[BaseModel]:
        class MyModel(BaseModel):
            array: NDArray[Shape[tuple(L[s] for s in shape)], dtype]

        return MyModel

    return _model


@pytest.fixture(scope="session")
def model_rgb() -> type[BaseModel]:
    class RGB(BaseModel):
        array: (
            NDArray[Shape[int, int], Number]
            | NDArray[Shape[int, int, L[3]], Number]
            | NDArray[Shape[int, int, L[3], L[4]], Number]
            | None
        ) = Field(None)

    return RGB


@pytest.fixture(scope="session")
def model_blank() -> type[BaseModel]:
    """A model with any shape and dtype"""

    class BlankModel(BaseModel):
        array: NDArray[Shape[*Repeat], Any]

    return BlankModel
