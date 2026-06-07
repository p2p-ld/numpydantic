"""
When used as a model field, checks against a numpy array
"""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray


class Foo(BaseModel):
    x: NDArray[Literal["3, 3, 3"], Any]


foo = Foo(x=np.zeros((3, 3, 3)))
