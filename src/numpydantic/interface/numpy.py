"""
Interface to numpy arrays
"""

from typing import Any

from numpydantic.interface.interface import Interface

try:
    from numpy import ndarray

    ENABLED = True

except ImportError:
    ENABLED = False
    ndarray = None


class NumpyInterface(Interface):
    """
    Numpy :class:`~numpy.ndarray` s!
    """

    input_types = (ndarray, list)
    return_type = ndarray

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check that this is in fact a numpy ndarray or something that can be
        coerced to one
        """
        if isinstance(array, ndarray):
            return True
        else:
            try:
                _ = ndarray(array)
                return True
            except TypeError:
                return False

    def before_validation(self, array: Any) -> ndarray:
        """
        Coerce to an ndarray. We have already checked if coercion is possible
        in :meth:`.check`
        """
        if not isinstance(array, ndarray):
            array = ndarray(array)
        return array

    @classmethod
    def enabled(cls) -> bool:
        """Check that numpy is present in the environment"""
        return ENABLED
