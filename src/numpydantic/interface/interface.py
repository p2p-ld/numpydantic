"""
Base Interface metaclass
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from operator import attrgetter
from typing import Any, Generic, Tuple, Type, TypedDict, TypeVar, Union

import numpy as np
from pydantic import SerializationInfo, TypeAdapter, ValidationError

from numpydantic.exceptions import (
    DtypeError,
    NoMatchError,
    ShapeError,
    TooManyMatchesError,
)
from numpydantic.shape import check_shape
from numpydantic.types import DtypeType, NDArrayType, ShapeType

T = TypeVar("T", bound=NDArrayType)


class InterfaceMark(TypedDict):
    """JSON-able mark to be able to round-trip json dumps"""

    module: str
    cls: str
    version: str


@dataclass(kw_only=True)
class JsonDict:
    """
    Representation of array when dumped with round_trip == True.

    Using a dataclass rather than a pydantic model to not tempt
    us to use more sophisticated types than can be serialized to json.
    """

    type: str

    @abstractmethod
    def to_array_input(self) -> Any:
        """
        Convert this roundtrip specifier to the relevant input class
        (one of the ``input_types`` of an interface).
        """

    def to_dict(self) -> dict:
        """
        Convenience method for casting dataclass to dict,
        removing None-valued items
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def get_adapter(cls) -> TypeAdapter:
        """Convenience method to get a typeadapter for this class"""
        return TypeAdapter(cls)

    @classmethod
    def is_valid(cls, val: dict) -> bool:
        """
        Check whether a given dictionary matches this JsonDict specification

        Returns:
            bool - true if valid, false if not
        """
        adapter = cls.get_adapter()
        try:
            _ = adapter.validate_python(val)
            return True
        except ValidationError:
            return False


class Interface(ABC, Generic[T]):
    """
    Abstract parent class for interfaces to different array formats
    """

    input_types: Tuple[Any, ...]
    return_type: Type[T]
    priority: int = 0

    def __init__(self, shape: ShapeType = Any, dtype: DtypeType = Any) -> None:
        self.shape = shape
        self.dtype = dtype

    def validate(self, array: Any) -> T:
        """
        Validate input, returning final array type

        Calls the methods, in order:

        * array = :meth:`.before_validation` (array)
        * dtype = :meth:`.get_dtype` (array) - get the dtype from the array,
            override if eg. the dtype is not contained in ``array.dtype``
        * valid = :meth:`.validate_dtype` (dtype) - check that the dtype matches
            the one in the NDArray specification. Override if special
            validation logic is needed for a given format
        * :meth:`.raise_for_dtype` (valid, dtype) - after checking dtype validity,
            raise an exception if it was invalid. Override to implement custom
            exceptions or error conditions, or make validation errors conditional.
        * array = :meth:`.after_validate_dtype` (array) - hook for additional
            validation or array modification mid-validation
        * shape = :meth:`.get_shape` (array) - get the shape from the array,
            override if eg. the shape is not contained in ``array.shape``
        * valid = :meth:`.validate_shape` (shape) - check that the shape matches
            the one in the NDArray specification. Override if special validation
            logic is needed.
        * :meth:`.raise_for_shape` (valid, shape) - after checking shape validity,
            raise an exception if it was invalid. You know the deal bc it's the same
            as raise for dtype.
        * :meth:`.after_validation` - hook after validation for modifying the array
            that is set as the model field value

        Follow the method signatures and return types to override.

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

        dtype = self.get_dtype(array)
        dtype_valid = self.validate_dtype(dtype)
        self.raise_for_dtype(dtype_valid, dtype)
        array = self.after_validate_dtype(array)

        shape = self.get_shape(array)
        shape_valid = self.validate_shape(shape)
        self.raise_for_shape(shape_valid, shape)

        array = self.after_validation(array)

        return array

    def before_validation(self, array: Any) -> NDArrayType:
        """
        Optional step pre-validation that coerces the input into a type that can be
        validated for shape and dtype

        Default method is a no-op
        """
        return array

    def get_dtype(self, array: NDArrayType) -> DtypeType:
        """
        Get the dtype from the input array
        """
        if hasattr(array.dtype, "type") and array.dtype.type is np.object_:
            return self.get_object_dtype(array)
        else:
            return array.dtype

    def get_object_dtype(self, array: NDArrayType) -> DtypeType:
        """
        When an array contains an object, get the dtype of the object contained
        by the array.
        """
        return type(array.ravel()[0])

    def validate_dtype(self, dtype: DtypeType) -> bool:
        """
        Validate the dtype of the given array, returning
        ``True`` if valid, ``False`` if not.
        """
        if self.dtype is Any:
            return True

        if isinstance(self.dtype, tuple):
            valid = dtype in self.dtype
        elif self.dtype is np.str_:
            valid = getattr(dtype, "type", None) in (np.str_, str) or dtype in (
                np.str_,
                str,
            )
        else:
            # try to match as any subclass, if self.dtype is a class
            try:
                valid = issubclass(dtype, self.dtype)
            except TypeError:
                # expected, if dtype or self.dtype is not a class
                valid = dtype == self.dtype

        return valid

    def raise_for_dtype(self, valid: bool, dtype: DtypeType) -> None:
        """
        After validating, raise an exception if invalid
        Raises:
            :class:`~numpydantic.exceptions.DtypeError`
        """
        if not valid:
            raise DtypeError(f"Invalid dtype! expected {self.dtype}, got {dtype}")

    def after_validate_dtype(self, array: NDArrayType) -> NDArrayType:
        """
        Hook to modify array after validating dtype.
        Default is a no-op.
        """
        return array

    def get_shape(self, array: NDArrayType) -> Tuple[int, ...]:
        """
        Get the shape from the array as a tuple of integers
        """
        return array.shape

    def validate_shape(self, shape: Tuple[int, ...]) -> bool:
        """
        Validate the shape of the given array against the shape
        specifier, returning ``True`` if valid, ``False`` if not.


        """
        if self.shape is Any:
            return True

        return check_shape(shape, self.shape)

    def raise_for_shape(self, valid: bool, shape: Tuple[int, ...]) -> None:
        """
        Raise a ShapeError if the shape is invalid.

        Raises:
            :class:`~numpydantic.exceptions.ShapeError`
        """
        if not valid:
            raise ShapeError(
                f"Invalid shape! expected shape {self.shape.prepared_args}, "
                f"got shape {shape}"
            )

    def after_validation(self, array: NDArrayType) -> T:
        """
        Optional step post-validation that coerces the intermediate array type into the
        return type

        Default method is a no-op
        """
        return array

    def mark_input(self, array: Any) -> Any:
        """
        Preserve metadata about the interface and passed input when dumping with
        ``round_trip``
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

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short name for this interface
        """

    @classmethod
    @abstractmethod
    def to_json(cls, array: Type[T], info: SerializationInfo) -> Union[list, JsonDict]:
        """
        Convert an array of :attr:`.return_type` to a JSON-compatible format using
        base python types
        """

    @classmethod
    def mark_json(cls, array: Union[list, dict]) -> dict:
        """
        When using ``model_dump_json`` with ``mark_interface: True`` in the ``context``,
        add additional annotations that would allow the serialized array to be
        roundtripped.

        Default is just to add an :class:`.InterfaceMark`

        Examples:

            >>> from pprint import pprint
            >>> pprint(Interface.mark_json([1.0, 2.0]))
            {'interface': {'cls': 'Interface',
                           'module': 'numpydantic.interface.interface',
                           'version': '1.2.2'},
             'value': [1.0, 2.0]}
        """
        return {"interface": cls.mark_interface(), "value": array}

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

    @classmethod
    def mark_interface(cls) -> InterfaceMark:
        """
        Create an interface mark indicating this interface for validation after
        JSON serialization with ``round_trip==True``
        """
        interface_module = inspect.getmodule(cls)
        interface_module = (
            None if interface_module is None else interface_module.__name__
        )
        try:
            v = (
                None
                if interface_module is None
                else version(interface_module.split(".")[0])
            )
        except PackageNotFoundError:
            v = None
        interface_name = cls.__name__
        return InterfaceMark(module=interface_module, cls=interface_name, version=v)
