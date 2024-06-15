"""
Base Interface metaclass
"""

from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Any, Generic, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import SerializationInfo

from numpydantic.exceptions import (
    DtypeError,
    NoMatchError,
    ShapeError,
    TooManyMatchesError,
)
from numpydantic.shape import check_shape
from numpydantic.types import DtypeType, NDArrayType, ShapeType

T = TypeVar("T", bound=NDArrayType)


class Interface(ABC, Generic[T]):
    """
    Abstract parent class for interfaces to different array formats
    """

    input_types: Tuple[Any, ...]
    return_type: Type[T]
    priority: int = 0

    def __init__(self, shape: ShapeType, dtype: DtypeType) -> None:
        self.shape = shape
        self.dtype = dtype

    def validate(self, array: Any) -> T:
        """
        Validate input, returning final array type

        Calls the methods, in order:

        * :meth:`.before_validation`
        * :meth:`.validate_dtype`
        * :meth:`.validate_shape`
        * :meth:`.after_validation`

        passing the ``array`` argument and returning it from each.

        Implementing an interface subclass largely consists of overriding these methods
        as needed.

        Raises:
            If validation fails, rather than eg. returning ``False``, exceptions will
            be raised (to halt the rest of the pydantic validation process).
            When using interfaces outside of pydantic, you must catch both
            :class:`.DtypeError` and :class:`.ShapeError` (both of which are children
            of :class:`.InterfaceError` )
        """
        array = self.before_validation(array)
        array = self.validate_dtype(array)
        array = self.validate_shape(array)
        array = self.after_validation(array)
        return array

    def before_validation(self, array: Any) -> NDArrayType:
        """
        Optional step pre-validation that coerces the input into a type that can be
        validated for shape and dtype

        Default method is a no-op
        """
        return array

    def validate_dtype(self, array: NDArrayType) -> NDArrayType:
        """
        Validate the dtype of the given array, returning it unmutated.

        Raises:
            :class:`~numpydantic.exceptions.DtypeError`
        """
        if self.dtype is Any:
            return array

        if isinstance(self.dtype, tuple):
            valid = array.dtype in self.dtype
        else:
            valid = array.dtype == self.dtype

        if not valid:
            raise DtypeError(f"Invalid dtype! expected {self.dtype}, got {array.dtype}")
        return array

    def validate_shape(self, array: NDArrayType) -> NDArrayType:
        """
        Validate the shape of the given array, returning it unmutated

        Raises:
            :class:`~numpydantic.exceptions.ShapeError`
        """
        if self.shape is Any:
            return array
        if not check_shape(array.shape, self.shape):
            raise ShapeError(
                f"Invalid shape! expected shape {self.shape.prepared_args}, "
                f"got shape {array.shape}"
            )
        return array

    def after_validation(self, array: NDArrayType) -> T:
        """
        Optional step post-validation that coerces the intermediate array type into the
        return type

        Default method is a no-op
        """
        return array

    @classmethod
    @abstractmethod
    def check(cls, array: Any) -> bool:
        """
        Method to check whether a given input applies to this interface
        """

    @classmethod
    @abstractmethod
    def enabled(cls) -> bool:
        """
        Check whether this array interface can be used (eg. its dependent packages are
        installed, etc.)
        """

    @classmethod
    def to_json(
        cls, array: Type[T], info: Optional[SerializationInfo] = None
    ) -> Union[list, dict]:
        """
        Convert an array of :attr:`.return_type` to a JSON-compatible format using
        base python types
        """
        if not isinstance(array, np.ndarray):  # pragma: no cover
            array = np.array(array)
        return array.tolist()

    @classmethod
    def interfaces(
        cls, with_disabled: bool = False, sort: bool = True
    ) -> Tuple[Type["Interface"], ...]:
        """
        Enabled interface subclasses

        Args:
            with_disabled (bool): If ``True`` , get every known interface.
                If ``False`` (default), get only enabled interfaces.
            sort (bool): If ``True`` (default), sort interfaces by priority.
                If ``False`` , sorted by definition order. Used for recursion:
                we only want to sort once at the top level.
        """
        # get recursively
        subclasses = []
        for i in cls.__subclasses__():
            if with_disabled:
                subclasses.append(i)

            if i.enabled():
                subclasses.append(i)

            subclasses.extend(i.interfaces(with_disabled=with_disabled, sort=False))

        if sort:
            subclasses = sorted(
                subclasses,
                key=attrgetter("priority"),
                reverse=True,
            )

        return tuple(subclasses)

    @classmethod
    def return_types(cls) -> Tuple[NDArrayType, ...]:
        """Return types for all enabled interfaces"""
        return tuple([i.return_type for i in cls.interfaces()])

    @classmethod
    def input_types(cls) -> Tuple[Any, ...]:
        """Input types for all enabled interfaces"""
        in_types = []
        for iface in cls.interfaces():
            if isinstance(iface.input_types, (tuple, list)):
                in_types.extend(iface.input_types)
            else:  # pragma: no cover
                in_types.append(iface.input_types)

        return tuple(in_types)

    @classmethod
    def match(cls, array: Any, fast: bool = False) -> Type["Interface"]:
        """
        Find the interface that should be used for this array based on its input type

        First runs the ``check`` method for all interfaces returned by
        :meth:`.Interface.interfaces` **except** for :class:`.NumpyInterface` ,
        and if no match is found then try the numpy interface. This is because
        :meth:`.NumpyInterface.check` can be expensive, as we could potentially
        try to

        Args:
            fast (bool): if ``False`` , check all interfaces and raise exceptions for
              having multiple matching interfaces (default). If ``True`` ,
              check each interface (as ordered by its ``priority`` , decreasing),
              and return on the first match.
        """
        # first try and find a non-numpy interface, since the numpy interface
        # will try and load the array into memory in its check method
        interfaces = cls.interfaces()
        non_np_interfaces = [i for i in interfaces if i.__name__ != "NumpyInterface"]
        np_interface = [i for i in interfaces if i.__name__ == "NumpyInterface"][0]

        if fast:
            matches = []
            for i in non_np_interfaces:
                if i.check(array):
                    return i
        else:
            matches = [i for i in non_np_interfaces if i.check(array)]

        if len(matches) > 1:
            msg = f"More than one interface matches input {array}:\n"
            msg += "\n".join([f"  - {i}" for i in matches])
            raise TooManyMatchesError(msg)
        elif len(matches) == 0:
            # now try the numpy interface
            if np_interface.check(array):
                return np_interface
            else:
                raise NoMatchError(f"No matching interfaces found for input {array}")
        else:
            return matches[0]

    @classmethod
    def match_output(cls, array: Any) -> Type["Interface"]:
        """
        Find the interface that should be used based on the output type -
        in the case that the output type differs from the input type, eg.
        the HDF5 interface, match an instantiated array for purposes of
        serialization to json, etc.
        """
        matches = [i for i in cls.interfaces() if isinstance(array, i.return_type)]
        if len(matches) > 1:
            msg = f"More than one interface matches output {array}:\n"
            msg += "\n".join([f"  - {i}" for i in matches])
            raise TooManyMatchesError(msg)
        elif len(matches) == 0:
            raise NoMatchError(f"No matching interfaces found for output {array}")
        else:
            return matches[0]
