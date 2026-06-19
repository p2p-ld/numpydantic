from typing import Annotated

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArraySchema, Shape


class MyModel(BaseModel):
    array: Annotated[np.ndarray, NDArraySchema(Shape(3, 3), np.uint8)]


class MyModel2(BaseModel):
    array: Annotated[np.ndarray, NDArraySchema(Shape[3, 3], np.uint8)]


instance = MyModel(array=np.ones((3, 3), np.uint8))

instance.array = instance.array + 3

assert isinstance(instance.array, np.ndarray)


def needs_array(x: np.ndarray) -> np.ndarray:
    return x


needs_array(instance.array)
