"""Interface-declared input types are allowed when used within a pydantic model"""

from typing import reveal_type

from pydantic import BaseModel

from numpydantic import NDArray
from numpydantic.interface.hdf5 import H5ArrayPath


class MyModel(BaseModel):

    array: NDArray
    something: int


instance = MyModel(array=H5ArrayPath("/tmp/test.h5", "some/dataset"), something=2)

reveal_type(instance)
reveal_type(instance.array)
reveal_type(MyModel)
reveal_type(H5ArrayPath("/tmp/test.h5", "some/dataset"))
