"""
Interface for Dask arrays
"""

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.interface.typing import ConstructorSpec, InterfaceTyping
from numpydantic.types import DtypeType, NDArrayType

try:
    from dask.array import from_array
    from dask.array.core import Array as DaskArray
except ImportError:  # pragma: no cover
    DaskArray = None


def _as_tuple(a_list: Any) -> tuple:
    """Make a list of list into a tuple of tuples"""
    return tuple(
        [_as_tuple(item) if isinstance(item, list) else item for item in a_list]
    )


class DaskJsonDict(JsonDict):
    """
    Round-trip json serialized form of a dask array
    """

    type: Literal["dask"]
    name: str
    chunks: Iterable[tuple[int, ...]]
    dtype: str
    shape: tuple[int, ...] | None = None
    value: list

    def to_array_input(self) -> DaskArray:
        """Construct a dask array"""
        np_array = np.array(self.value, dtype=self.dtype)
        if self.shape is not None and np_array.shape != self.shape:
            np_array = self.reshape_input(np_array, self.shape)
        array = from_array(
            np_array,
            name=self.name,
            chunks=_as_tuple(self.chunks),
        )
        return array


class DaskTyping(InterfaceTyping):
    """Static-typing companion for :class:`DaskInterface`."""

    constructors = (
        ConstructorSpec(fullname="dask.array.zeros"),
        ConstructorSpec(fullname="dask.array.ones"),
        ConstructorSpec(fullname="dask.array.empty"),
        ConstructorSpec(fullname="dask.array.full"),
    )

    @classmethod
    def emit_imports(cls) -> list[str]:
        """import dask.array and numpy"""
        return ["import dask.array", "import numpy"]

    @classmethod
    def emit_constructor_source(cls, shape: tuple[int, ...], dtype: str) -> str | None:
        """render a call to dask.array.zeros"""
        return f"dask.array.zeros({tuple(shape)!r}, dtype={dtype})"


class DaskInterface(Interface):
    """
    Interface for Dask :class:`~dask.array.core.Array`
    """

    name = "dask"
    input_types = (DaskArray,)
    return_type = DaskArray
    json_model = DaskJsonDict
    typing = DaskTyping

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        check if array is a dask array
        """
        if DaskArray is None:  # pragma: no cover - no tests for interface deps atm
            return False
        elif isinstance(array, DaskArray):
            return True
        elif isinstance(array, dict):
            return DaskJsonDict.is_valid(array)
        else:
            return False

    def before_validation(self, array: DaskArray) -> NDArrayType:
        """
        Try and coerce dicts that should be model objects into the model objects
        """
        try:
            if issubclass(self.dtype, BaseModel):
                flat_array = array.reshape(-1)
                if len(flat_array) == 0:
                    return array

                if isinstance(flat_array[0].compute(), dict):

                    def _chunked_to_model(array: np.ndarray) -> np.ndarray:
                        def _vectorized_to_model(
                            item: dict | BaseModel,
                        ) -> BaseModel:
                            if not isinstance(item, self.dtype):
                                return self.dtype(**item)
                            else:  # pragma: no cover
                                return item

                        return np.vectorize(_vectorized_to_model)(array)

                    array = array.map_blocks(_chunked_to_model, dtype=self.dtype)
        except TypeError:
            # fine, dtype isn't a type
            pass
        return array

    def get_object_dtype(self, array: NDArrayType) -> DtypeType:
        """Dask arrays require a compute() call to retrieve a single value"""
        flat_array = array.reshape(-1)
        if len(flat_array) == 0:
            return Any
        else:
            return type(flat_array[0].compute())

    @classmethod
    def enabled(cls) -> bool:
        """check if we successfully imported dask"""
        return DaskArray is not None

    @classmethod
    def to_json(
        cls, array: DaskArray, info: SerializationInfo | None = None
    ) -> list | DaskJsonDict:
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
        np_array = np.array(array)
        as_json = np_array.tolist()
        if not isinstance(as_json, list):
            as_json = [as_json]
        if info.round_trip:
            as_json = DaskJsonDict(
                type=cls.name,
                value=as_json,
                name=array.name,
                chunks=array.chunks,
                dtype=str(np_array.dtype),
                shape=array.shape,
            )
        return as_json
