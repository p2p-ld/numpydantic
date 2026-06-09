from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.interface.hdf5 import H5ArrayPath


class MyModel(BaseModel):
    array: NDArray[Shape[1, 2, 3]]


instance = MyModel(array=H5ArrayPath("./example.h5", "/some/dataset"))
