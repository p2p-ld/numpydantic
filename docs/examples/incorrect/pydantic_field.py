from pydantic import BaseModel

from numpydantic import NDArray, Shape
from numpydantic.interface.hdf5 import H5ArrayPath


class MyModel(BaseModel):
    array: NDArray[Shape[1, 2, 3]]


# you can't just use any old thing
x = MyModel(array="./example.h5")
y = MyModel(array=5)


# and the annotation refuses to accept non-array inputs
# when used in non-pydantic contexts
def passthrough(array: NDArray[Shape[1, 2, 3]]) -> NDArray[Shape[1, 2, 3]]:
    return array


z = passthrough(H5ArrayPath("./example.h5", "/some/dataset"))
