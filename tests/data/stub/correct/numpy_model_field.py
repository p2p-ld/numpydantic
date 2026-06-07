"""
When used as a model field, checks against a numpy array
"""

from typing import Any
from typing import Literal as L

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray


class Foo(BaseModel):
    x: NDArray[tuple[L[3], L[3], L[3]], Any]


foo = Foo(x=np.zeros((3, 3, 3)))
