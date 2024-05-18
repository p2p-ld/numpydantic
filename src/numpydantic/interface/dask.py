"""
Interface for Dask arrays
"""

from typing import Any, Optional

import numpy as np
from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface

try:
    from dask.array.core import Array as DaskArray
except ImportError:  # pragma: no cover
    DaskArray = None


class DaskInterface(Interface):
    """
    Interface for Dask :class:`~dask.array.core.Array`
    """

    input_types = (DaskArray,)
    return_type = DaskArray

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        check if array is a dask array
        """
        if DaskArray is not None and isinstance(array, DaskArray):
            return True
        return False

    @classmethod
    def enabled(cls) -> bool:
        """check if we successfully imported dask"""
        return DaskArray is not None

    @classmethod
    def to_json(
        cls, array: DaskArray, info: Optional[SerializationInfo] = None
    ) -> list:
        """
        Convert an array to a JSON serializable array by first converting to a numpy
        array and then to a list.

        .. note::

            This is likely a very memory intensive operation if you are using dask for
            large arrays. This can't be avoided, since the creation of the json string
            happens in-memory with Pydantic, so you are likely looking for a different
            method of serialization here using the python object itself rather than
            its JSON representation.
        """
        return np.array(array).tolist()
