from typing import Any, Callable, Optional, Tuple, Type, Union

import numpy as np
import pytest
from pydantic import BaseModel, Field

from numpydantic import NDArray, Shape
from numpydantic.dtype import Number


@pytest.fixture(scope="function")
def array_model() -> (
    Callable[[Tuple[int, ...], Union[Type, np.dtype]], Type[BaseModel]]
):
    def _model(
        shape: Tuple[int, ...] = (10, 10), dtype: Union[Type, np.dtype] = float
    ) -> Type[BaseModel]:
        shape_str = ", ".join([str(s) for s in shape])

        class MyModel(BaseModel):
            array: NDArray[Shape[shape_str], dtype]

        return MyModel

    return _model


@pytest.fixture(scope="session")
def model_rgb() -> Type[BaseModel]:
    class RGB(BaseModel):
        array: Optional[
            Union[
                NDArray[Shape["* x, * y"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b"], Number],
                NDArray[Shape["* x, * y, 3 r_g_b, 4 r_g_b_a"], Number],
            ]
        ] = Field(None)

    return RGB


@pytest.fixture(scope="session")
def model_blank() -> Type[BaseModel]:
    """A model with any shape and dtype"""

    class BlankModel(BaseModel):
        array: NDArray[Shape["*, ..."], Any]

    return BlankModel
