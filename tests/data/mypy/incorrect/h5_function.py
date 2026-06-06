"""Interface-declared input types are NOT allowed outside a pydantic model"""

from numpydantic import NDArray
from numpydantic.interface.hdf5 import H5ArrayPath


def main(array: NDArray) -> NDArray:
    return array


array = main(array=H5ArrayPath("/tmp/test.h5", "some/dataset"))
