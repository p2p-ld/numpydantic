"""
A :class:`pydantic.GetPydanticCoreSchema` - like type annotation
to make type checkers happy :)
"""

from typing import Any

from pydantic import GetPydanticSchema

from numpydantic import NDArray
from numpydantic.types import DtypeType
from numpydantic.validation.shape import Shape


def NDArraySchema(
    shape: type[Shape] | Shape | str | tuple = Shape, dtype: DtypeType = Any
) -> GetPydanticSchema:
    """
    Specify shape and dtype constraints in an :class:`typing.Annotated` type.

    In addition to validating dtype and shape constraints,
    the ``type`` of the array will also be validated -
    i.e. if the annotation is for a :class:`numpy.ndarray`,
    a :class:`dask.array.Array` will be rejected
    even if it has the correct shape and dtype.

    Examples:

        >>> from typing import Annotated as A
        >>> from numpydantic import Shape, NDArraySchema
        >>> import numpy as np
        >>> from pydantic import BaseModel

        >>> class MyModel(BaseModel):
        >>>     array: A[np.ndarray, NDArraySchema(Shape(3, 3), np.uint8)]

        or, without Shape

        >>> class MyOtherModel(BaseModel):
        >>>     array: A[np.ndarray, NDArraySchema((3, 3), np.uint8)]

        Valid:

        >>> MyModel(array=np.ones((3, 3), dtype=np.uint8))

        Not valid:

        >>> MyModel(array=dask.array.ones((3, 3), dtype=np.uint8))

    Args:
        shape (Shape | str | tuple): The shape specification, either as a Shape class,
            or as the shape constraint string/tuple by itself.
        dtype:

    Returns:

    """
    if shape is not Shape and shape is not Any and not issubclass(shape, Shape):
        shape = Shape[shape]

    ndarray = NDArray[shape, dtype]

    return GetPydanticSchema(
        ndarray.__get_pydantic_core_schema__, ndarray.__get_pydantic_json_schema__
    )
