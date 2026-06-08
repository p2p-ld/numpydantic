"""
Interface to numpy arrays
"""

import contextlib
from typing import Any, Literal

from pydantic import BaseModel, SerializationInfo

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.interface.typing import ConstructorSpec, InterfaceTyping

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
    # allow shape to be None for backwards compatibility.
    shape: tuple[int, ...] | None = None
    # store absolute python identifier for objects
    object_cls: str | None = None

    def to_array_input(self) -> ndarray:
        """
        Construct a numpy array
        """
        array = np.array(self.value, dtype=self.dtype)

        # recast to object, if relevant
        if self.dtype == "object" and self.object_cls is not None:
            array = self.cast_objects(array, self.object_cls)

        if self.shape is not None and array.shape != self.shape:
            array = self.reshape_input(array, self.shape)
        return array


class NumpyTyping(InterfaceTyping):
    """Static-typing companion for :class:`NumpyInterface`."""

    constructors = (
        ConstructorSpec(fullname="numpy.ones"),
        ConstructorSpec(fullname="numpy.zeros"),
        ConstructorSpec(fullname="numpy.empty"),
        ConstructorSpec(fullname="numpy.full"),
        # Newer numpy stubs route the public ``np.zeros`` etc. through a
        # ``Final[_ConstructorEmpty]`` protocol instance, so mypy sees the
        # call as a method on that protocol.
        ConstructorSpec(
            fullname="numpy._core.multiarray._ConstructorEmpty.__call__",
            mode="method",
        ),
    )

    @classmethod
    def emit_imports(cls) -> list[str]:
        """Just importing numpy over here!"""
        return ["import numpy"]

    @classmethod
    def emit_constructor_source(cls, shape: tuple[int, ...], dtype: str) -> str | None:
        """Constructor using :func:`numpy.zeros`"""
        return f"numpy.zeros({tuple(shape)!r}, dtype={dtype})"


class NumpyInterface(Interface):
    """
    Numpy :class:`~numpy.ndarray` s!
    """

    name = "numpy"
    input_types = (ndarray,)
    return_type = ndarray
    json_model = NumpyJsonDict
    priority = -999
    """
    The numpy interface is usually the interface of last resort.
    We want to use any more specific interface that we might have,
    because the numpy interface checks for anything that could be coerced
    to a numpy array (see :meth:`.NumpyInterface.check` )
    """
    typing = NumpyTyping

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check that this is in fact a numpy ndarray or something that can be
        coerced to one
        """
        if array is None:
            return False

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
        if not isinstance(array, ndarray):
            array = np.array(array)

        try:
            # try to convert a dict to a basemodel, if relevant
            # this is the *only* dtype coercion that we should attempt to do,
            # because pydantic treats dicts as equivalent to models in inputs.
            # other coercion when e.g. deserializing from JSON should go
            # in the JSONDict object's deserialization methods.
            if (
                issubclass(self.dtype, BaseModel)
                and len(array) > 0
                and isinstance(array.flat[0], dict)
            ):
                array = np.vectorize(lambda x: self.dtype(**x))(array)
        except TypeError:
            # fine, dtype isn't a type
            pass

        return array

    @classmethod
    def enabled(cls) -> bool:
        """Check that numpy is present in the environment"""
        return ENABLED

    @classmethod
    def to_json(cls, array: ndarray, info: SerializationInfo = None) -> list | JsonDict:
        """
        Convert an array of :attr:`.return_type` to a JSON-compatible format using
        base python types
        """
        if not isinstance(array, np.ndarray):  # pragma: no cover
            array = np.array(array)

        json_array = [array.tolist()] if array.ndim == 0 else array.tolist()

        if info.round_trip:
            # store object dtype
            dtype = str(array.dtype)
            object_cls = None
            if dtype == "object":
                with contextlib.suppress(AttributeError, IndexError):
                    obj = array.ravel()[0].__class__
                    object_cls = f"{obj.__module__}.{obj.__name__}"

            json_array = NumpyJsonDict(
                type=cls.name,
                dtype=dtype,
                value=json_array,
                shape=array.shape,
                object_cls=object_cls,
            )
        return json_array
