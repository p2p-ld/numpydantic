"""
Interface to numpy arrays
"""

from typing import Any

from numpydantic.interface.interface import Interface

try:
    import numpy as np
    from numpy import ndarray

    ENABLED = True

except ImportError:  # pragma: no cover
    ENABLED = False
    ndarray = None
    np = None


class NumpyInterface(Interface):
    """
    Numpy :class:`~numpy.ndarray` s!
    """

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
        if not isinstance(array, ndarray):
            array = np.array(array)
        return array

    @classmethod
    def enabled(cls) -> bool:
        """Check that numpy is present in the environment"""
        return ENABLED
