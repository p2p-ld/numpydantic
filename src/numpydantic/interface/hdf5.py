"""
Interfaces for HDF5 Datasets
"""

import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple, Union

import numpy as np
from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface
from numpydantic.types import NDArrayType

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

if sys.version_info.minor >= 10:
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

H5Arraylike: TypeAlias = Tuple[Union[Path, str], str]


class H5ArrayPath(NamedTuple):
    """Location specifier for arrays within an HDF5 file"""

    file: Union[Path, str]
    """Location of HDF5 file"""
    path: str
    """Path within the HDF5 file"""


class H5Proxy:
    """
    Proxy class to mimic numpy-like array behavior with an HDF5 array

    The attribute and item access methods only open the file for the duration of the
    method, making it less perilous to share this object between threads and processes.

    This class attempts to be a passthrough class to a :class:`h5py.Dataset` object,
    including its attributes and item getters/setters.

    When using read-only methods, no locking is attempted (beyond the HDF5 defaults),
    but when using the write methods (setting an array value), try and use the
    ``locking`` methods of :class:`h5py.File` .

    Args:
        file (pathlib.Path | str): Location of hdf5 file on filesystem
        path (str): Path to array within hdf5 file
    """

    def __init__(self, file: Union[Path, str], path: str):
        self._h5f = None
        self.file = Path(file)
        self.path = path

    def array_exists(self) -> bool:
        """Check that there is in fact an array at :attr:`.path` within :attr:`.file`"""
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj is not None

    @classmethod
    def from_h5array(cls, h5array: H5ArrayPath) -> "H5Proxy":
        """Instantiate using :class:`.H5ArrayPath`"""
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

    def open(self, mode: str = "r") -> "h5py.Dataset":
        """
        Return the opened :class:`h5py.Dataset` object

        You must remember to close the associated file with :meth:`.close`
        """
        if self._h5f is None:
            self._h5f = h5py.File(self.file, mode)
        return self._h5f.get(self.path)

    def close(self) -> None:
        """
        Close the :class:`h5py.File` object left open when returning the dataset with
        :meth:`.open`
        """
        if self._h5f is not None:
            self._h5f.close()
        self._h5f = None


class H5Interface(Interface):
    """
    Interface for Arrays stored as datasets within an HDF5 file.

    Takes a :class:`.H5ArrayPath` specifier to select a :class:`h5py.Dataset` from a
    :class:`h5py.File` and returns a :class:`.H5Proxy` class that acts like a
    passthrough numpy-like interface to the dataset.
    """

    input_types = (
        H5ArrayPath,
        H5Arraylike,
    )
    return_type = H5Proxy

    @classmethod
    def enabled(cls) -> bool:
        """Check whether h5py can be imported"""
        return h5py is not None

    @classmethod
    def check(cls, array: Union[H5ArrayPath, Tuple[Union[Path, str], str]]) -> bool:
        """
        Check that the given array is a :class:`.H5ArrayPath` or something that
        resembles one.
        """
        if isinstance(array, H5ArrayPath):
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
        if isinstance(array, H5ArrayPath):
            array = H5Proxy.from_h5array(h5array=array)
        elif isinstance(array, (tuple, list)) and len(array) == 2:  # pragma: no cover
            array = H5Proxy(file=array[0], path=array[1])
        else:  # pragma: no cover
            # this should never happen really since `check` confirms this before
            # we'd reach here, but just to complete the if else...
            raise ValueError(
                "Need to specify a file and a path within an HDF5 file to use the HDF5 "
                "Interface"
            )

        if not array.array_exists():
            raise ValueError(
                f"HDF5 file located at {array.file}, "
                f"but no array found at {array.path}"
            )

        return array

    @classmethod
    def to_json(cls, array: H5Proxy, info: Optional[SerializationInfo] = None) -> dict:
        """
        Dump to a dictionary containing

        * ``file``: :attr:`.file`
        * ``path``: :attr:`.path`
        * ``attrs``: Any HDF5 attributes on the dataset
        * ``array``: The array as a list of lists
        """
        try:
            dset = array.open()
            meta = {
                "file": array.file,
                "path": array.path,
                "attrs": dict(dset.attrs),
                "array": dset[:].tolist(),
            }
            return meta
        finally:
            array.close()
