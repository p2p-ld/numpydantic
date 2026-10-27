"""
When used as a model field, checks against a numpy array
"""

from typing import Any, assert_type
from typing import Literal as L

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray, Shape, unshape


class Foo(BaseModel):
    x: NDArray[Shape[L[3], L[3], L[3]], Any]


foo = Foo(x=unshape(np.zeros((3, 3, 3))))
assert_type(foo.x, NDArray[Shape[L[3], L[3], L[3]], Any])
