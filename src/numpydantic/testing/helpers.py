from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, computed_field

from numpydantic import NDArray, Shape
from numpydantic.dtype import Float
from numpydantic.interface import Interface
from numpydantic.types import DtypeType, NDArrayType


class InterfaceCase(ABC):
    """
    An interface test helper that allows a given interface to generate and validate
    arrays in one of its formats.

    Each instance of "interface test case" should be considered one of the
    potentially multiple realizations of a given interface.
    If an interface has multiple formats (eg. zarr's different `store` s),
    then it should have several test helpers.
    """

    @property
    @abstractmethod
    def interface(self) -> Interface:
        """The interface that this helper is for"""

    @classmethod
    def array_from_case(
        cls, case: "ValidationCase", path: Optional[Path] = None
    ) -> Optional[NDArrayType]:
        """
        Generate an array from the given validation case.

        Returns ``None`` if an array can't be generated for a specific case.
        """
        return cls.make_array(shape=case.shape, dtype=case.dtype, path=path)

    @classmethod
    @abstractmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
    ) -> Optional[NDArrayType]:
        """
        Make an array from a shape and dtype, and a path if needed
        """

    @classmethod
    def validate_case(cls, case: "ValidationCase", path: Path) -> bool:
        """
        Validate a generated array against the annotation in the validation case.

        Kept in the InterfaceCase in case an interface has specific
        needs aside from just validating against a model, but typically left as is.

        If an array can't be generated for a given case, returns `None`
        so that the calling function can know to skip rather than fail the case.

        Raises exceptions if validation fails (or succeeds when it shouldn't)

        Args:
            case (ValidationCase): The validation case to validate.
            path (Path): Path to generate arrays into, if any.

        Returns:
            ``True`` if array is valid and was supposed to be,
                or invalid and wasn't supposed to be
        """
        import pytest

        array = cls.array_from_case(case, path)
        if array is None:
            pytest.skip()
        if case.passes:
            case.model(array=array)
            return True
        else:
            with pytest.raises(ValidationError):
                case.model(array=array)
            return True

    @classmethod
    def skip(cls, shape: Tuple[int, ...], dtype: DtypeType) -> bool:
        """
        Whether a given interface should be skipped for the case
        """
        # Assume an interface case is valid for all other cases
        return False


class ValidationCase(BaseModel):
    """
    Test case for validating an array.

    Contains both the validating model and the parameterization for an array to
    test in a given interface
    """

    id: Optional[str] = None
    """
    String identifying the validation case
    """
    annotation: Any = NDArray[Shape["10, 10, *"], Float]
    """
    Array annotation used in the validating model
    Any typed because the types of type annotations are weird
    """
    shape: Tuple[int, ...] = (10, 10, 10)
    """Shape of the array to validate"""
    dtype: Union[Type, np.dtype] = float
    """Dtype of the array to validate"""
    passes: bool = False
    """Whether the validation should pass or not"""
    interface: Optional[InterfaceCase] = None
    """The interface test case to generate and validate the array with"""
    path: Optional[Path] = None
    """The path to generate arrays into, if any."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field()
    def model(self) -> Type[BaseModel]:
        """A model with a field ``array`` with the given annotation"""
        annotation = self.annotation

        class Model(BaseModel):
            array: annotation

        return Model

    def validate_case(self, path: Optional[Path] = None) -> bool:
        """
        Whether the generated array correctly validated against the annotation,
        given the interface

        Args:
            path (:class:`pathlib.Path`): Directory to generate array into, if on disk.

        Raises:
            ValueError: if an ``interface`` is missing
        """
        if self.interface is None:
            raise ValueError("Missing an interface")
        if path is None:
            if self.path:
                path = self.path
            else:
                raise ValueError("Missing a path to generate arrays into")

        return self.interface.validate_case(self, path)

    def array(self, path: Path) -> NDArrayType:
        """Generate an array for the validation case if we have an interface to do so"""
        if self.interface is None:
            raise ValueError("Missing an interface")
        if path is None:
            if self.path:
                path = self.path
            else:
                raise ValueError("Missing a path to generate arrays into")

        return self.interface.array_from_case(self, path)

    def merge(
        self, other: Union["ValidationCase", Sequence["ValidationCase"]]
    ) -> "ValidationCase":
        """
        Merge two validation cases

        Dump both, excluding any unset fields, and merge, preferring `other`.

        ``valid`` is ``True`` if and only if it is ``True`` in both.
        """
        if isinstance(other, Sequence):
            return merge_cases(self, *other)

        self_dump = self.model_dump(exclude_unset=True)
        other_dump = other.model_dump(exclude_unset=True)

        # dumps might not have set `valid`, use only the ones that have
        valids = [
            v
            for v in [self_dump.get("valid", None), other_dump.get("valid", None)]
            if v is not None
        ]
        valid = all(valids)

        # combine ids if present
        ids = "-".join(
            [
                str(v)
                for v in [self_dump.get("id", None), other_dump.get("id", None)]
                if v is not None
            ]
        )

        merged = {**self_dump, **other_dump}
        merged["valid"] = valid
        merged["id"] = ids
        return ValidationCase(**merged)

    def skip(self) -> bool:
        """
        Whether this case should be skipped
        (eg. due to the interface case being incompatible
        with the requested dtype or shape)
        """
        return bool(
            self.interface is not None and self.interface.skip(self.shape, self.dtype)
        )


def merge_cases(*args: ValidationCase) -> ValidationCase:
    """
    Merge multiple validation cases
    """
    if len(args) == 1:
        return args[0]

    case = args[0]
    for arg in args[1:]:
        case = case.merge(arg)
    return case
