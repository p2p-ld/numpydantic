from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Any, Generic, List, Type, TypeVar, Tuple

from nptyping.shape_expression import check_shape

from numpydantic.exceptions import DtypeError, ShapeError
from numpydantic.types import DtypeType, NDArrayType, ShapeType

T = TypeVar("T", bound=NDArrayType)


class Interface(ABC, Generic[T]):
    """
    Abstract parent class for interfaces to different array formats
    """

    return_type: Type[T]
    priority: int = 0

    def __init__(self, shape: ShapeType, dtype: DtypeType) -> None:
        self.shape = shape
        self.dtype = dtype

    def validate(self, array: Any) -> T:
        """
        Validate input, returning final array type
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
        if not array.dtype == self.dtype:
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
                f"Invalid shape! expected shape {self.shape.prepared_args}, got shape {array.shape}"
            )
        return array

    def after_validation(self, array: NDArrayType) -> T:
        """
        Optional step post-validation that coerces the intermediate array type into the return type

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
        Check whether this array interface can be used (eg. its dependent packages are installed, etc.)
        """

    @classmethod
    def interfaces(cls) -> Tuple[Type["Interface"], ...]:
        """
        Enabled interface subclasses
        """
        return tuple(
            sorted(
                [i for i in Interface.__subclasses__() if i.enabled()],
                key=attrgetter("priority"),
                reverse=True,
            )
        )

    @classmethod
    def array_types(cls) -> Tuple[NDArrayType, ...]:
        """Return types for all enabled interfaces"""
        return tuple([i.return_type for i in cls.interfaces()])

    @classmethod
    def match(cls, array: Any) -> Type["Interface"]:
        """
        Find the interface that should be used for this array
        """
        matches = [i for i in cls.interfaces() if i.check(array)]
        if len(matches) > 1:
            msg = f"More than one interface matches input {array}:\n"
            msg += "\n".join([f"  - {i}" for i in matches])
            raise ValueError(msg)
        elif len(matches) == 0:
            raise ValueError(f"No matching interfaces found for input {array}")
        else:
            return matches[0]
