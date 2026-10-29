"""
Validating arrays of any supported type against a shape and dtype
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic_core.core_schema import ValidationInfo

from numpydantic.exceptions import DtypeError, ShapeError
from numpydantic.interface import Interface
from numpydantic.shape._typevars import typevar_positions
from numpydantic.types import DtypeType, NDArrayType, ShapeType

SHAPE_VARIABLES_KEY = "__npd_typevars__"
"""
The key in a model's ``__dict__`` that holds the size of each of its shape variables.
pydantic leaves it out of dumps, ``repr``, equality, and iteration.
"""


def validate_array(
    value: Any,
    shape: ShapeType,
    dtype: DtypeType,
    typevar_sizes: Mapping[str, int] | None = None,
) -> NDArrayType:
    """
    Validate an array with the matching :class:`.Interface`,
    returning the validated array.

    Args:
        value: The array, or anything an interface accepts as input
        shape: The normalized shape to validate against
        dtype: The dtype to validate against
        typevar_sizes: Sizes that the shape variables in ``shape`` already have
    """
    interface = Interface.match(value)(shape, dtype, typevar_sizes=typevar_sizes)
    try:
        return interface.validate(value)
    except (ShapeError, DtypeError) as e:
        if isinstance(value, (str, Path)):
            raise type(e)(
                f"{e}\nThe passed input looks like a path, make sure you have "
                "the required dependencies installed to load the array!"
            ) from e
        raise


def get_validate_interface(
    shape: ShapeType, dtype: DtypeType, cross_field_validation: bool = False
) -> Callable[[Any, ValidationInfo], NDArrayType]:
    """
    A pydantic validator for arrays with a shape and dtype,
    using :func:`.validate_array`

    Shape variables are always matched within one array.
    With ``cross_field_validation``, they also have to have the same sizes
    in all the fields of a model, which are kept in its ``__dict__``.
    """
    positions = typevar_positions(shape) if cross_field_validation else {}

    def validate_interface(value: Any, info: ValidationInfo) -> NDArrayType:
        # info.data is None outside of models, like with validate_call
        if not positions or info.data is None:
            return validate_array(value, shape, dtype)

        # The fields of one model share info.data, which becomes its __dict__.
        # The sizes are changed in place, because assignment validates against
        # a copy of __dict__ and then builds the new __dict__ from the old one,
        # and this dict is the one object they share.
        sizes = info.data.setdefault(SHAPE_VARIABLES_KEY, {})
        value = validate_array(value, shape, dtype, sizes)
        array_shape = value.shape
        sizes.update(
            {
                name: array_shape[index]
                for index, name in positions.items()
                if index < len(array_shape)
            }
        )
        return value

    return validate_interface
