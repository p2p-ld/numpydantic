from numpydantic.ndarray_generic import NDArray
from pydantic import BaseModel
from typing import Literal as L
import numpy as np


class MyClass(BaseModel):
    array: NDArray[L[4, 5], int]


model = MyClass(array=np.array([1, 2, 3, 4]))

model2 = MyClass(array=np.array([1, 2, 3]))

model3 = MyClass(array=(1, 2))


array: NDArray[L[4], np.int64] = np.array([1, 2, 3])
array2: NDArray[L[3], np.int64] = [1, 2, 3]
