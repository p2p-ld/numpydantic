"""
Ensure that mypy doesn't complain about slicing for ndarray types
Basic test that the protocol class is type checking like a protocol class

https://github.com/p2p-ld/numpydantic/issues/57
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel

from numpydantic import NDArray


class Foo(BaseModel):
    x: NDArray[Literal["3"], np.float64]

    def bar(self) -> None:
        print(self.x[0])
        print(self.x[1])
        print(self.x[0:2])
        print(self.x[0:1, 0:1])
        print(self.x[0:1, 0:1, 0:1])


foo = Foo(x=np.zeros((3, 3, 3)))
foo.bar()
