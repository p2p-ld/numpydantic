"""
Interface to zarr arrays
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface

try:
    import zarr
    from zarr.core import Array as ZarrArray
    from zarr.storage import StoreLike
except ImportError:  # pragma: no cover
    ZarrArray = None
    StoreLike = None
    storage = None


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


class ZarrInterface(Interface):
    """
    Interface to in-memory or on-disk zarr arrays
    """

    input_types = (Path, ZarrArray, ZarrArrayPath)
    return_type = ZarrArray

    @classmethod
    def enabled(cls) -> bool:
        """True if zarr is installed"""
        return ZarrArray is not None

    @staticmethod
    def _get_array(
        array: Union[ZarrArray, str, Path, ZarrArrayPath, Sequence]
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

    @classmethod
    def to_json(
        cls,
        array: Union[ZarrArray, str, Path, ZarrArrayPath, Sequence],
        info: Optional[SerializationInfo] = None,
    ) -> dict:
        """
        Dump just the metadata for an array from :meth:`zarr.core.Array.info_items`
        plus the :meth:`zarr.core.Array.hexdigest`.

        The full array can be returned by passing ``'zarr_dump_array': True`` to the
        serialization ``context`` ::

            model.model_dump_json(context={'zarr_dump_array': True})
        """
        dump_array = False
        if info is not None and info.context is not None:
            dump_array = info.context.get("zarr_dump_array", False)

        array = cls._get_array(array)
        info = array.info_items()
        info_dict = {i[0]: i[1] for i in info}
        info_dict["hexdigest"] = array.hexdigest()

        if dump_array:
            info_dict["array"] = array[:].tolist()

        return info_dict
