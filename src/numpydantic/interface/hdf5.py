from pathlib import Path
from typing import Any, NamedTuple, Tuple, Union, TypeAlias

import numpy as np

from numpydantic.interface.interface import Interface
from numpydantic.types import NDArrayType

try:
    import h5py
except ImportError:
    h5py = None

H5Arraylike: TypeAlias = Tuple[Union[Path, str], str]


class H5Array(NamedTuple):
    """Location specifier for arrays within an HDF5 file"""

    file: Union[Path, str]
    """Location of HDF5 file"""
    path: str
    """Path within the HDF5 file"""


class H5Proxy:
    """
    Proxy class to mimic numpy-like array behavior with an HDF5 array

    The attribute and item access methods only open the file for the duration of the method,
    making it less perilous to share this object between threads and processes.

    This class attempts to be a passthrough class to a :class:`h5py.Dataset` object,
    including its attributes and item getters/setters.

    When using read-only methods, no locking is attempted (beyond the HDF5 defaults),
    but when using the write methods (setting an array value), try and use the ``locking``
    methods of :class:`h5py.File` .

    Args:
        file (pathlib.Path | str): Location of hdf5 file on filesystem
        path (str): Path to array within hdf5 file
    """

    def __init__(self, file: Union[Path, str], path: str):
        self.file = Path(file)
        self.path = path

    def array_exists(self) -> bool:
        """Check that there is in fact an array at :attr:`.path` within :attr:`.file`"""
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj is not None

    @classmethod
    def from_h5array(cls, h5array: H5Array) -> "H5Proxy":
        """Instantiate using :class:`.H5Array`"""
        return H5Proxy(file=h5array.file, path=h5array.path)

    def __getattr__(self, item: str):
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return getattr(obj, item)

    def __getitem__(self, item: Union[int, slice]) -> np.ndarray:
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj[item]

    def __setitem__(self, key: Union[int, slice], value: Union[int, float, np.ndarray]):
        with h5py.File(self.file, "r+", locking=True) as h5f:
            obj = h5f.get(self.path)
            obj[key] = value


class H5Interface(Interface):
    """
    Interface for Arrays stored as datasets within an HDF5 file.

    Takes a :class:`.H5Array` specifier to select a :class:`h5py.Dataset` from a
    :class:`h5py.File` and returns a :class:`.H5Proxy` class that acts like a
    passthrough numpy-like interface to the dataset.
    """

    input_types = (
        H5Array,
        H5Arraylike,
    )
    return_type = H5Proxy

    @classmethod
    def enabled(cls) -> bool:
        """Check whether h5py can be imported"""
        return h5py is not None

    @classmethod
    def check(cls, array: Union[H5Array, Tuple[Union[Path, str], str]]) -> bool:
        """Check that the given array is a :class:`.H5Array` or something that resembles one."""
        if isinstance(array, H5Array):
            return True

        if isinstance(array, (tuple, list)) and len(array) == 2:
            # check that the first arg is an hdf5 file
            try:
                file = Path(array[0])
            except TypeError:
                # not a path, we don't apply.
                return False

            if not file.exists():
                return False

            # hdf5 files are commonly given odd suffixes,
            # so we just try and open it and see what happens
            try:
                with h5py.File(file, "r"):
                    # don't check that the array exists and raise here,
                    # this check is just for whether the validator applies or not.
                    pass
                return True
            except (FileNotFoundError, OSError):
                return False

        return False

    def before_validation(self, array: Any) -> NDArrayType:
        """Create an :class:`.H5Proxy` to use throughout validation"""
        if isinstance(array, H5Array):
            array = H5Proxy.from_h5array(h5array=array)
        elif isinstance(array, (tuple, list)) and len(array) == 2:
            array = H5Proxy(file=array[0], path=array[1])
        else:
            raise ValueError(
                "Need to specify a file and a path within an HDF5 file to use the HDF5 Interface"
            )

        if not array.array_exists():
            raise ValueError(
                f"HDF5 file located at {array.file}, "
                f"but no array found at {array.path}"
            )

        return array
