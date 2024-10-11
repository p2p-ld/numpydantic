"""
Interface to zarr arrays
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.types import DtypeType

try:
    import zarr
    from numcodecs import VLenUTF8
    from zarr.core import Array as ZarrArray
    from zarr.storage import StoreLike
except ImportError:  # pragma: no cover
    ZarrArray = None
    StoreLike = None
    storage = None
    VLenUTF8 = None


@dataclass
class ZarrArrayPath:
    """
    Map to an array within a zarr store.

    See :func:`zarr.open`
    """

    file: Union[Path, str]
    """Location of Zarr store file or directory"""
    path: Optional[str] = None
    """Path to array within hierarchical zarr store"""

    def open(self, **kwargs: dict) -> ZarrArray:
        """Open the zarr array at the provided path"""
        return zarr.open(str(self.file), path=self.path, **kwargs)

    @classmethod
    def from_iterable(cls, spec: Sequence) -> "ZarrArrayPath":
        """
        Construct a :class:`.ZarrArrayPath` specifier from an iterable,
        rather than kwargs
        """
        if len(spec) == 1:
            return ZarrArrayPath(file=spec[0])
        elif len(spec) == 2:
            return ZarrArrayPath(file=spec[0], path=spec[1])
        else:
            raise ValueError("Only len 1-2 iterables can be used for a ZarrArrayPath")


class ZarrJsonDict(JsonDict):
    """Round-trip Json-able version of a Zarr Array"""

    info: dict[str, str]
    type: Literal["zarr"]
    file: Optional[str] = None
    path: Optional[str] = None
    dtype: Optional[str] = None
    value: Optional[list] = None

    def to_array_input(self) -> Union[ZarrArray, ZarrArrayPath]:
        """
        Construct a ZarrArrayPath if file and path are present,
        otherwise a ZarrArray
        """
        if self.file:
            array = ZarrArrayPath(file=self.file, path=self.path)
        else:
            array = zarr.array(self.value, dtype=self.dtype)
        return array


class ZarrInterface(Interface):
    """
    Interface to in-memory or on-disk zarr arrays
    """

    name = "zarr"
    input_types = (Path, ZarrArray, ZarrArrayPath)
    return_type = ZarrArray
    json_model = ZarrJsonDict

    @classmethod
    def enabled(cls) -> bool:
        """True if zarr is installed"""
        return ZarrArray is not None

    @staticmethod
    def _get_array(
        array: Union[ZarrArray, str, dict, ZarrJsonDict, Path, ZarrArrayPath, Sequence]
    ) -> ZarrArray:
        if isinstance(array, ZarrArray):
            return array

        if isinstance(array, (str, Path)):
            array = ZarrArrayPath(file=array)
        elif isinstance(array, (tuple, list)):
            array = ZarrArrayPath.from_iterable(array)

        return array.open(mode="a")

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check if array is in-memory zarr array,
        a path to a zarr array, or a :class:`.ZarrArrayPath`
        """
        if isinstance(array, ZarrArray):
            return True

        if isinstance(array, dict):
            if array.get("type", False) == cls.name:
                return True
            # continue checking if dict contains a zarr file
            array = array.get("file", "")

        # See if can be coerced to ZarrArrayPath
        if isinstance(array, (Path, str)):
            array = ZarrArrayPath(file=array)

        if isinstance(array, (tuple, list)):
            # something that can be coerced to ZarrArrayPath
            with contextlib.suppress(ValueError):
                array = ZarrArrayPath.from_iterable(array)

        if isinstance(array, ZarrArrayPath):
            with contextlib.suppress(Exception):
                arr = array.open(mode="r")
                if isinstance(arr, ZarrArray):
                    return True

        return False

    def before_validation(
        self, array: Union[ZarrArray, str, Path, ZarrArrayPath, Sequence]
    ) -> ZarrArray:
        """
        Ensure that the zarr array is opened
        """
        return self._get_array(array)

    def get_dtype(self, array: ZarrArray) -> DtypeType:
        """
        Override base dtype getter to handle zarr's string-as-object encoding.
        """
        if (
            getattr(array.dtype, "type", None) is np.object_
            and array.filters
            and any([isinstance(f, VLenUTF8) for f in array.filters])
        ):
            return np.str_
        else:
            return array.dtype

    @classmethod
    def to_json(
        cls,
        array: Union[ZarrArray, str, Path, ZarrArrayPath, Sequence],
        info: Optional[SerializationInfo] = None,
    ) -> Union[list, ZarrJsonDict]:
        """
        Dump a Zarr Array to JSON

        If ``info.round_trip == False``, dump the array as a list of lists.
        This may be a memory-intensive operation.

        Otherwise, dump the metadata for an array from
        :meth:`zarr.core.Array.info_items`
        plus the :meth:`zarr.core.Array.hexdigest` as a :class:`.ZarrJsonDict`

        If either the ``dump_array`` value in the context dictionary is ``True``
        or the zarr array is an in-memory array, dump the array as well
        (since without a persistent array it would be impossible to roundtrip and
        dumping to JSON would be meaningless)

        Passing ```dump_array': True`` to the serialization ``context``
        looks like this::

            model.model_dump_json(context={'zarr_dump_array': True})
        """
        array = cls._get_array(array)

        if info.round_trip:
            dump_array = False
            if info is not None and info.context is not None:
                dump_array = info.context.get("dump_array", False)
            is_file = False

            as_json = {"type": cls.name}
            as_json["dtype"] = array.dtype.name
            if hasattr(array.store, "dir_path"):
                is_file = True
                as_json["file"] = array.store.dir_path()
                as_json["path"] = array.name
            as_json["info"] = {i[0]: i[1] for i in array.info_items()}
            as_json["info"]["hexdigest"] = array.hexdigest()

            if dump_array or not is_file:
                as_json["value"] = array[:].tolist()

            as_json = ZarrJsonDict(**as_json)
        else:
            as_json = array[:].tolist()

        return as_json
