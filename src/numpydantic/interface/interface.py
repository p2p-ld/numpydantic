"""
Base Interface metaclass
"""

import builtins
import importlib
import inspect
import sys
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, Union

if TYPE_CHECKING:
    from numpydantic.interface.typing import InterfaceTyping

import numpy as np
from pydantic import BaseModel, SerializationInfo, ValidationError

from numpydantic.exceptions import (
    DtypeError,
    MarkMismatchError,
    NoMatchError,
    ShapeError,
    TooManyMatchesError,
)
from numpydantic.types import DtypeType, NDArrayType, ShapeType
from numpydantic.validation import validate_dtype, validate_shape

T = TypeVar("T", bound=NDArrayType)
U = TypeVar("U", bound="JsonDict")
V = TypeVar("V")  # input type
W = TypeVar("W")  # Any type in handle_input


class InterfaceMark(BaseModel):
    """JSON-able mark to be able to round-trip json dumps"""

    module: str
    cls: str
    name: str
    version: str

    def is_valid(self, cls: type["Interface"], raise_on_error: bool = False) -> bool:
        """
        Check that a given interface matches the mark.

        Args:
            cls (Type): Interface type to check
            raise_on_error (bool): Raise an ``MarkMismatchError`` when the match
                is incorrect

        Returns:
            bool

        Raises:
            :class:`.MarkMismatchError` if requested by ``raise_on_error``
            for an invalid match
        """
        mark = cls.mark_interface()
        valid = self == mark
        if not valid and raise_on_error:
            raise MarkMismatchError(
                "Mismatch between serialized mark and current interface, "
                f"Serialized: {self}; current: {cls}"
            )
        return valid

    def match_by_name(self) -> type["Interface"] | None:
        """
        Try to find a matching interface by its name, returning it if found,
        or None if not found.
        """
        for i in Interface.interfaces(sort=False):
            if i.name == self.name:
                return i
        return None


class JsonDict(BaseModel):
    """
    Representation of array when dumped with round_trip == True.

    .. admonition:: Developer's Note

        Any JsonDict that contains an actual array should be named ``value``
        rather than array (or any other name), and nothing but the
        array data should be named ``value`` .

        During JSON serialization, it becomes ambiguous what contains an array
        of data vs. an array of metadata. For the moment we would like to
        reserve the ability to have lists of metadata, so until we rule that out,
        we would like to be able to avoid iterating over every element of an array
        in any context parameter transformation like relativizing/absolutizing paths.
        To avoid that, it's good to agree on a single value name -- ``value`` --
        and avoid using it for anything else.

    """

    type: str

    @abstractmethod
    def to_array_input(self) -> V:
        """
        Convert this roundtrip specifier to the relevant input class
        (one of the ``input_types`` of an interface).
        """

    @classmethod
    def is_valid(cls, val: dict, raise_on_error: bool = False) -> bool:
        """
        Check whether a given dictionary matches this JsonDict specification

        Args:
            val (dict): The dictionary to check for validity
            raise_on_error (bool): If ``True``, raise the validation error
                rather than returning a bool. (default: ``False``)

        Returns:
            bool - true if valid, false if not
        """
        try:
            _ = cls.model_validate(val)
            return True
        except ValidationError as e:
            if raise_on_error:
                raise e
            return False

    @classmethod
    def handle_input(cls: type[U], value: dict | U | W) -> V | W:
        """
        Handle input that is the json serialized roundtrip version
        (from :func:`~pydantic.BaseModel.model_dump` with ``round_trip=True``)
        converting it to the input format with :meth:`.JsonDict.to_array_input`
        or passing it through if not applicable
        """
        if isinstance(value, dict):
            value = cls(**value).to_array_input()
        elif isinstance(value, cls):
            value = value.to_array_input()
        return value

    @staticmethod
    def reshape_input(value: T, shape: tuple[int, ...]) -> T:
        """
        If a `reshape` value is present on the array, and the array shape doesn't match,
        attempt to reshape it.
        """
        if value.shape != shape:
            try:
                value = value.reshape(shape)
            except ValueError:
                warnings.warn(
                    f"Input data has shape {value.shape}, "
                    f"but roundtrip form specifies {shape},"
                    f"and {value.shape} can't be cast to {shape}. "
                    f"Attempting to proceed with validation without reshaping.",
                    stacklevel=1,
                )
        return value

    @staticmethod
    def resolve_python_identifier(ref: str) -> Any:
        """
        Given some fully-qualified package.subpackage.Class identifier,
        return the referenced object, importing if needed.
        """
        if "." not in ref:
            return getattr(builtins, ref)
        else:
            module_name, obj = ref.rsplit(".", 1)
            module = sys.modules.get(module_name, importlib.import_module(module_name))

            return getattr(module, obj)

    def cast_objects(self, array: T, object_cls_name: str) -> T:
        """
        Recast objects in object arrays to the type they were before serialization
        """
        if object_cls_name == "datetime.datetime":
            # special case: must use constructor method
            array = np.vectorize(lambda x: datetime.fromisoformat(x))(array)
        else:
            object_cls = self.resolve_python_identifier(object_cls_name)
            if isinstance(object_cls, type) and issubclass(object_cls, BaseModel):
                # mild code duplication but we want both -
                # convert back to proper object type when deserializing from JSON,
                # and also coerce dicts to objects when given on object instantiation
                array = np.vectorize(lambda x: object_cls(**x))(array)
            else:
                array = np.vectorize(lambda x: object_cls(x))(array)
        return array


class MarkedJson(BaseModel):
    """
    Model of JSON dumped with an additional interface mark
    with ``model_dump_json({'mark_interface': True})``
    """

    interface: InterfaceMark
    value: list | dict
    """
    Inner value of the array, we don't validate for JsonDict here, 
    that should be downstream from us for performance reasons 
    """

    @classmethod
    def try_cast(cls, value: V | dict) -> Union[V, "MarkedJson"]:
        """
        Try to cast to MarkedJson if applicable, otherwise return input
        """
        if isinstance(value, dict) and "interface" in value and "value" in value:
            try:
                value = MarkedJson(**value)
            except ValidationError:
                # fine, just not a MarkedJson dict even if it looks like one
                return value
        return value


class Interface(ABC, Generic[T]):
    """
    Abstract parent class for interfaces to different array formats
    """

    input_types: tuple[Any, ...]
    return_type: type[T]
    priority: int = 0
    typing: ClassVar[type["InterfaceTyping"] | None] = None
    """
    Optional static-typing companion class used by the mypy plugin and
    the mypy test generator. ``None`` means this interface does not opt
    into static constructor inference.
    """

    def __init__(self, shape: ShapeType = Any, dtype: DtypeType = Any) -> None:
        self.shape = shape
        self.dtype = dtype

    def validate(self, array: Any) -> T:
        """
        Validate input, returning final array type

        Calls the methods, in order:

        * array = :meth:`.deserialize` (array)
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

        If validation fails, rather than eg. returning ``False``, exceptions will
        be raised (to halt the rest of the pydantic validation process).
        When using interfaces outside of pydantic, you must catch both
        :class:`.DtypeError` and :class:`.ShapeError` (both of which are children
        of :class:`.InterfaceError` )

        Raises:
            :class:`.DtypeError`: Dtype of data doesn't match specification
            :class:`.ShapeError`: Shape of data doesn't match specification

        """
        array = self.deserialize(array)

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

    def deserialize(self, array: Any) -> V | Any:
        """
        If given a JSON serialized version of the array,
        deserialize it first.

        If a roundtrip-serialized :class:`.JsonDict`,
        pass to :meth:`.JsonDict.handle_input`.

        If a roundtrip-serialized :class:`.MarkedJson`,
        unpack mark, check for validity, warn if not,
        and try to continue with validation
        """
        if isinstance(marked_array := MarkedJson.try_cast(array), MarkedJson):
            try:
                marked_array.interface.is_valid(self.__class__, raise_on_error=True)
            except MarkMismatchError as e:
                warnings.warn(
                    str(e) + "\nAttempting to continue validation...", stacklevel=2
                )
            array = marked_array.value

        return self.json_model.handle_input(array)

    def before_validation(self, array: Any) -> NDArrayType:
        """
        Optional step pre-validation that coerces the input into a type that can be
        validated for shape and dtype

        Default method is a no-op
        """
        return array

    def get_dtype(self, array: NDArrayType) -> DtypeType:
        """
        Get the dtype from the input array.
        """
        if hasattr(array.dtype, "type") and array.dtype.type is np.object_:
            return self.get_object_dtype(array)
        else:
            return array.dtype

    def get_object_dtype(self, array: NDArrayType) -> DtypeType:
        """
        When an array contains an object, get the dtype of the object contained
        by the array.

        If this method returns `Any`, the dtype validation passes -
        used for e.g. empty arrays for which the dtype of the array can't be determined
        (since there are no objects).
        """
        try:
            return type(array.ravel()[0])
        except IndexError:
            return Any

    def validate_dtype(self, dtype: DtypeType) -> bool:
        """
        Validate the dtype of the given array, returning
        ``True`` if valid, ``False`` if not.
        """
        return validate_dtype(dtype, self.dtype)

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

    def get_shape(self, array: NDArrayType) -> tuple[int, ...]:
        """
        Get the shape from the array as a tuple of integers
        """
        return array.shape

    def validate_shape(self, shape: tuple[int, ...]) -> bool:
        """
        Validate the shape of the given array against the shape
        specifier, returning ``True`` if valid, ``False`` if not.


        """
        if self.shape is Any:
            return True

        return validate_shape(shape, self.shape)

    def raise_for_shape(self, valid: bool, shape: tuple[int, ...]) -> None:
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

    @property
    @abstractmethod
    def json_model(self) -> JsonDict:
        """
        The :class:`.JsonDict` model used for roundtripping
        JSON serialization
        """

    @classmethod
    @abstractmethod
    def to_json(cls, array: type[T], info: SerializationInfo) -> list | JsonDict:
        """
        Convert an array of :attr:`.Interface.return_type` to a JSON-compatible format
        using base python types
        """

    @classmethod
    def mark_json(cls, array: list | dict) -> dict:
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
    ) -> tuple[type["Interface"], ...]:
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
    def return_types(cls) -> tuple[NDArrayType, ...]:
        """Return types for all enabled interfaces"""
        return tuple([i.return_type for i in cls.interfaces()])

    @classmethod
    def input_types(cls) -> tuple[Any, ...]:
        """Input types for all enabled interfaces"""
        in_types = []
        for iface in cls.interfaces():
            if isinstance(iface.input_types, (tuple, list)):
                in_types.extend(iface.input_types)
            else:  # pragma: no cover
                in_types.append(iface.input_types)

        return tuple(in_types)

    @classmethod
    def match_mark(cls, array: Any) -> type["Interface"] | None:
        """
        Match a marked JSON dump of this array to the interface that it indicates.

        First find an interface that matches by name, and then run its
        ``check`` method, because arrays can be dumped with a mark
        but without ``round_trip == True`` (and thus can't necessarily
        use the same interface that they were dumped with)

        Returns:
            Interface if match found, None otherwise
        """
        mark = MarkedJson.try_cast(array)
        if not isinstance(mark, MarkedJson):
            return None

        interface = mark.interface.match_by_name()
        if interface is not None and interface.check(mark.value):
            return interface
        return None

    @classmethod
    def match(cls, array: Any, fast: bool = False) -> type["Interface"]:
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
        # Shortcircuit match if this is a marked json dump
        array = MarkedJson.try_cast(array)
        if (match := cls.match_mark(array)) is not None:
            return match
        elif isinstance(array, MarkedJson):
            array = array.value

        # first try and find a non-numpy interface, since the numpy interface
        # will try and load the array into memory in its check method
        interfaces = cls.interfaces()
        non_np_interfaces = [i for i in interfaces if i.name != "numpy"]
        np_interface = [i for i in interfaces if i.name == "numpy"][0]

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
    def match_output(cls, array: Any) -> type["Interface"]:
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
    @lru_cache(maxsize=32)
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
        except (
            PackageNotFoundError
        ):  # pragma: no cover - no tests for missing interface deps
            v = None

        return InterfaceMark(
            module=interface_module, cls=cls.__name__, name=cls.name, version=v
        )


class Proxy(ABC):
    """
    A proxy class that exposes some non-array data source (like a video) as an array
    """

    @classmethod
    @abstractmethod
    def proxy_for(cls) -> type[Interface]:
        """
        Declare the interface that this is a proxy for,
        allowing the proxy to be used with the NDArraySchema annotation
        with any of the input types that the Interface supports.
        """
        raise NotImplementedError()
