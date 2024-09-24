"""
Interfaces for HDF5 Datasets

.. note::

    HDF5 arrays are accessed through a proxy class :class:`.H5Proxy` .
    Getting/setting values should work as normal, **except** that setting
    values on nested views is impossible - 
    
    Specifically this doesn't work:
    
    .. code-block:: python
    
        my_model.array[0][0] = 1
    
    But this does work:
    
    .. code-block:: python
    
        my_model.array[0,0] = 1
        
    To have direct access to the hdf5 dataset, use the
    :meth:`.H5Proxy.open` method.
    
Datetimes 
---------

Datetimes are supported as a dtype annotation, but currently they must be stored
as ``S32`` isoformatted byte strings (timezones optional) like:    

.. code-block:: python

    import h5py
    from datetime import datetime
    import numpy as np
    data = np.array([datetime.now().isoformat().encode('utf-8')], dtype="S32")
    h5f = h5py.File('test.hdf5', 'w')
    h5f.create_dataset('data', data=data)
    
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.types import DtypeType, NDArrayType

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

if sys.version_info.minor >= 10:
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

H5Arraylike: TypeAlias = Tuple[Union[Path, str], str]

T = TypeVar("T")


class H5ArrayPath(NamedTuple):
    """Location specifier for arrays within an HDF5 file"""

    file: Union[Path, str]
    """Location of HDF5 file"""
    path: str
    """Path within the HDF5 file"""
    field: Optional[Union[str, List[str]]] = None
    """Refer to a specific field within a compound dtype"""


class H5JsonDict(JsonDict):
    """Round-trip Json-able version of an HDF5 dataset"""

    file: str
    path: str
    field: Optional[str] = None

    def to_array_input(self) -> H5ArrayPath:
        """Construct an :class:`.H5ArrayPath`"""
        return H5ArrayPath(
            **{k: v for k, v in self.model_dump().items() if k in H5ArrayPath._fields}
        )


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
        field (str, list[str]): Optional - refer to a specific field within
            a compound dtype
        annotation_dtype (dtype): Optional - the dtype of our type annotation
    """

    def __init__(
        self,
        file: Union[Path, str],
        path: str,
        field: Optional[Union[str, List[str]]] = None,
        annotation_dtype: Optional[DtypeType] = None,
    ):
        self._h5f = None
        self.file = Path(file).resolve()
        self.path = path
        self.field = field
        self._annotation_dtype = annotation_dtype
        self._h5arraypath = H5ArrayPath(self.file, self.path, self.field)

    def array_exists(self) -> bool:
        """Check that there is in fact an array at :attr:`.path` within :attr:`.file`"""
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj is not None

    @classmethod
    def from_h5array(cls, h5array: H5ArrayPath) -> "H5Proxy":
        """Instantiate using :class:`.H5ArrayPath`"""
        return H5Proxy(file=h5array.file, path=h5array.path, field=h5array.field)

    @property
    def dtype(self) -> np.dtype:
        """
        Get dtype of array, using :attr:`.field` if present
        """
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            if self.field is None:
                return obj.dtype
            else:
                return obj.dtype[self.field]

    def __array__(self) -> np.ndarray:
        """To a numpy array"""
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj[:]

    def __getattr__(self, item: str):
        if item == "__name__":
            # special case for H5Proxies that don't refer to a real file during testing
            return "H5Proxy"
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            val = getattr(obj, item)
            return val

    def __getitem__(
        self, item: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> Union[np.ndarray, DtypeType]:
        with h5py.File(self.file, "r") as h5f:
            obj = h5f.get(self.path)
            # handle compound dtypes
            if self.field is not None:
                # handle compound string dtype
                if encoding := h5py.h5t.check_string_dtype(obj.dtype[self.field]):
                    if isinstance(item, tuple):
                        item = (*item, self.field)
                    else:
                        item = (item, self.field)

                    try:
                        # single string
                        val = obj[item].decode(encoding.encoding)
                        if self._annotation_dtype is np.datetime64:
                            return np.datetime64(val)
                        else:
                            return val
                    except AttributeError:
                        # numpy array of bytes
                        val = np.char.decode(obj[item], encoding=encoding.encoding)
                        if self._annotation_dtype is np.datetime64:
                            return val.astype(np.datetime64)
                        else:
                            return val
                # normal compound type
                else:
                    obj = obj.fields(self.field)
            else:
                if h5py.h5t.check_string_dtype(obj.dtype):
                    obj = obj.asstr()

            val = obj[item]
            if self._annotation_dtype is np.datetime64:
                if isinstance(val, str):
                    return np.datetime64(val)
                else:
                    return val.astype(np.datetime64)
            else:
                return val

    def __setitem__(
        self,
        key: Union[int, slice, Tuple[Union[int, slice], ...]],
        value: Union[int, float, datetime, np.ndarray],
    ):
        # TODO: Make a generalized value serdes system instead of ad-hoc type conversion
        value = self._serialize_datetime(value)
        with h5py.File(self.file, "r+", locking=True) as h5f:
            obj = h5f.get(self.path)
            if self.field is None:
                obj[key] = value
            else:
                if isinstance(key, tuple):
                    key = (*key, self.field)
                    obj[key] = value
                else:
                    obj[key, self.field] = value

    def __len__(self) -> int:
        """self.shape[0]"""
        return self.shape[0]

    def __eq__(self, other: "H5Proxy") -> bool:
        """
        Check that we are referring to the same hdf5 array
        """
        if isinstance(other, H5Proxy):
            return self._h5arraypath == other._h5arraypath
        else:
            raise ValueError("Can only compare equality of two H5Proxies")

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

    def _serialize_datetime(self, v: Union[T, datetime]) -> Union[T, bytes]:
        """
        Convert a datetime into a bytestring
        """
        if self._annotation_dtype is np.datetime64:
            if not isinstance(v, Iterable):
                v = [v]
            v = np.array(v).astype("S32")
        return v


class H5Interface(Interface):
    """
    Interface for Arrays stored as datasets within an HDF5 file.

    Takes a :class:`.H5ArrayPath` specifier to select a :class:`h5py.Dataset` from a
    :class:`h5py.File` and returns a :class:`.H5Proxy` class that acts like a
    passthrough numpy-like interface to the dataset.
    """

    name = "hdf5"
    input_types = (H5ArrayPath, H5Arraylike, H5Proxy)
    return_type = H5Proxy
    json_model = H5JsonDict

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
        if isinstance(array, (H5ArrayPath, H5Proxy)):
            return True

        if isinstance(array, dict):
            if array.get("type", False) == cls.name:
                return True
            # continue checking if dict contains an hdf5 file
            file = array.get("file", "")
            array = (file, "")

        if isinstance(array, (tuple, list)) and len(array) in (2, 3):
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
        elif isinstance(array, H5Proxy):
            # nothing to do, already proxied
            pass
        elif isinstance(array, (tuple, list)) and len(array) == 2:  # pragma: no cover
            array = H5Proxy(file=array[0], path=array[1])
        elif isinstance(array, (tuple, list)) and len(array) == 3:
            array = H5Proxy(file=array[0], path=array[1], field=array[2])
        else:  # pragma: no cover
            # this should never happen really since `check` confirms this before
            # we'd reach here, but just to complete the if else...
            raise ValueError(
                "Need to specify a file and a path within an HDF5 file to use the HDF5 "
                "Interface"
            )
        array._annotation_dtype = self.dtype

        if not array.array_exists():
            raise ValueError(
                f"HDF5 file located at {array.file}, "
                f"but no array found at {array.path}"
            )

        return array

    def get_dtype(self, array: NDArrayType) -> DtypeType:
        """
        Get the dtype from the input array

        Subclasses to correctly handle
        """
        if h5py.h5t.check_string_dtype(array.dtype):
            # check for datetimes
            try:
                if array[0].dtype.type is np.datetime64:
                    return np.datetime64
                else:
                    return str
            except (AttributeError, TypeError):  # pragma: no cover
                # it's not a datetime, but it is some kind of string
                return str
            except (IndexError, ValueError):
                # if the dataset is empty, we can't tell if something is a datetime
                # or not, so we just tell the validation method what it wants to hear
                if self.dtype in (np.datetime64, str):
                    return self.dtype
                else:
                    return str
        else:
            return array.dtype

    @classmethod
    def to_json(cls, array: H5Proxy, info: Optional[SerializationInfo] = None) -> dict:
        """
        Render HDF5 array as JSON

        If ``round_trip == True``, we dump just the proxy info, a dictionary like:

        * ``file``: :attr:`.file`
        * ``path``: :attr:`.path`
        * ``attrs``: Any HDF5 attributes on the dataset
        * ``array``: The array as a list of lists

        Otherwise, we dump the array as a list of lists
        """
        if info.round_trip:
            as_json = {
                "type": cls.name,
            }
            as_json.update(array._h5arraypath._asdict())
        else:
            try:
                dset = array.open()
                as_json = dset[:].tolist()
            finally:
                array.close()

        return as_json
