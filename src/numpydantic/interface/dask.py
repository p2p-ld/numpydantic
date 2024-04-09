from typing import Any
from numpydantic.interface.interface import Interface

try:
    from dask.array.core import Array as DaskArray
except ImportError:
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
