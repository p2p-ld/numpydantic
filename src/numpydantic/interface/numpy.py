"""
Interface to numpy arrays
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.serialization import pydantic_serializer

try:
    import numpy as np
    from numpy import ndarray

    ENABLED = True

except ImportError:  # pragma: no cover
    ENABLED = False
    ndarray = None
    np = None


class NumpyJsonDict(JsonDict):
    """
    JSON-able roundtrip representation of numpy array
    """

    type: Literal["numpy"]
    dtype: str
    value: list

    def to_array_input(self) -> ndarray:
        """
        Construct a numpy array
        """
        return np.array(self.value, dtype=self.dtype)


class SerializableNDArray(np.ndarray):
    """
    Trivial subclass of :class:`numpy.ndarray` that allows
    an additional ``__pydantic_serializer__`` attr to allow it to be
    json roundtripped without the help of an interface

    References:
        https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    def __new__(cls, input_array: np.ndarray, **kwargs: dict[str, Any]):
        """Create a new ndarray instance, adding a new attribute"""
        obj = np.asarray(input_array, **kwargs).view(cls)
        obj.__pydantic_serializer__ = pydantic_serializer
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        self.__pydantic_serializer__ = getattr(obj, "__pydantic_serializer__", None)


class NumpyInterface(Interface):
    """
    Numpy :class:`~numpy.ndarray` s!
    """

    name = "numpy"
    input_types = (ndarray, list)
    return_type = ndarray
    json_model = NumpyJsonDict
    priority = -999
    """
    The numpy interface is usually the interface of last resort.
    We want to use any more specific interface that we might have,
    because the numpy interface checks for anything that could be coerced
    to a numpy array (see :meth:`.NumpyInterface.check` )
    """

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check that this is in fact a numpy ndarray or something that can be
        coerced to one
        """
        if array is None:
            return False

        if isinstance(array, (ndarray, SerializableNDArray)):
            return True
        elif isinstance(array, dict):
            return NumpyJsonDict.is_valid(array)
        else:
            try:
                _ = np.array(array)
                return True
            except Exception:
                return False

    def before_validation(self, array: Any) -> ndarray:
        """
        Coerce to an ndarray. We have already checked if coercion is possible
        in :meth:`.check`
        """
        if not isinstance(array, SerializableNDArray):
            array = SerializableNDArray(array)

        try:
            if issubclass(self.dtype, BaseModel) and isinstance(array.flat[0], dict):
                array = SerializableNDArray(
                    np.vectorize(lambda x: self.dtype(**x))(array)
                )
        except TypeError:
            # fine, dtype isn't a type
            pass

        return array

    @classmethod
    def enabled(cls) -> bool:
        """Check that numpy is present in the environment"""
        return ENABLED

    @classmethod
    def to_json(
        cls, array: ndarray, info: SerializationInfo = None
    ) -> Union[list, JsonDict]:
        """
        Convert an array of :attr:`.return_type` to a JSON-compatible format using
        base python types
        """
        if not isinstance(array, np.ndarray):  # pragma: no cover
            array = np.array(array)

        json_array = array.tolist()

        if info.round_trip:
            json_array = NumpyJsonDict(
                type=cls.name, dtype=str(array.dtype), value=json_array
            )
        return json_array
