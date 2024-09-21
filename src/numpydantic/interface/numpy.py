"""
Interface to numpy arrays
"""

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict

try:
    import numpy as np
    from numpy import ndarray

    ENABLED = True

except ImportError:  # pragma: no cover
    ENABLED = False
    ndarray = None
    np = None


@dataclass
class NumpyJsonDict(JsonDict):
    """
    JSON-able roundtrip representation of numpy array
    """

    type: Literal["numpy"]
    dtype: str
    array: list

    def to_array_input(self) -> ndarray:
        """
        Construct a numpy array
        """
        return np.array(self.array, dtype=self.dtype)


class NumpyInterface(Interface):
    """
    Numpy :class:`~numpy.ndarray` s!
    """

    name = "numpy"
    input_types = (ndarray, list)
    return_type = ndarray
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
        if isinstance(array, ndarray):
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
        if isinstance(array, dict):
            array = NumpyJsonDict(**array).to_array_input()

        if not isinstance(array, ndarray):
            array = np.array(array)
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
                type=cls.name, dtype=str(array.dtype), array=json_array
            )
        return json_array
