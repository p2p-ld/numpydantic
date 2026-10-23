import sys
from collections.abc import Callable
from typing import Any
from typing import Literal as L

import numpy as np
import pytest
from pydantic import BaseModel, Field

from numpydantic import AnyDims, NDArray, Shape
from numpydantic.dtype import Number

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack


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
        array: NDArray[Shape[Unpack[AnyDims]], Any]

    return BlankModel
