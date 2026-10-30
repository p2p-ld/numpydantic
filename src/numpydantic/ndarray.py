"""
An array type for pydantic, with shape and dtype validation and JSON-Schema
serialization

.. note::

    This module should *only* have the :class:`.NDArray` class in it, because the
    type stub ``ndarray.pyi`` is only created for :class:`.NDArray` . Otherwise,
    type checkers will complain about using any helper functions elsewhere -
    those all belong in :mod:`numpydantic.schema` .

    :class:`.NDArrayAlias` is in this module even though it's excluded
    from the type stub.

"""

from __future__ import annotations

import types
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema

from numpydantic.dtype import DType
from numpydantic.exceptions import InterfaceError, InvalidShapeExpressionError
from numpydantic.interface import Interface
from numpydantic.maps import python_to_nptyping
from numpydantic.schema import make_core_schema, make_ndarray_json_schema
from numpydantic.types import NDArrayType
from numpydantic.validation.array import validate_array
from numpydantic.validation.dtype import is_union

if TYPE_CHECKING:  # pragma: no cover
    from numpydantic import Shape


def _get_shape(dtype_candidate: Any) -> Shape:
    """
    Normalize the shape argument of ``NDArray[shape, dtype]`` with :class:`.Shape`
    """
    from numpydantic.shape import Repeat, Shape

    if dtype_candidate is Any or dtype_candidate is Shape or dtype_candidate is Repeat:
        shape = Any
    elif get_origin(dtype_candidate) is Literal:
        shape_expression = dtype_candidate.__args__[0]
        shape = Shape[shape_expression]
    elif get_origin(dtype_candidate) is tuple:
        shape = Shape[get_args(dtype_candidate)]
    elif isinstance(dtype_candidate, (str, tuple)):
        shape = Shape[dtype_candidate]
    else:
        raise InvalidShapeExpressionError(
            f"Unexpected argument '{dtype_candidate}', expecting"
            " Shape[...], tuple[...], or typing.Any."
        )
    return shape


def _get_dtype(dtype_candidate: Any) -> DType:
    """
    Override of base _get_dtype method to allow for compound tuple types
    """
    if dtype_candidate in python_to_nptyping:
        dtype_candidate = python_to_nptyping[dtype_candidate]
    is_dtype = isinstance(dtype_candidate, type) and issubclass(
        dtype_candidate, np.generic
    )

    if dtype_candidate is Any:
        dtype = Any
    elif is_dtype or is_union(dtype_candidate):
        dtype = dtype_candidate
    elif isinstance(dtype_candidate, tuple):  # pragma: no cover
        dtype = tuple([_get_dtype(dt) for dt in dtype_candidate])
    else:
        # arbitrary dtype - allow failure elsewhere :)
        dtype = dtype_candidate

    return dtype


class NDArrayAlias(types.GenericAlias):
    """
    ``NDArray[shape, dtype]``, with the normalized shape and dtype as its ``__args__``.

    Calling it validates an array, and ``isinstance`` checks whether an array is valid.
    """

    def __call__(self, value: Any) -> NDArrayType:
        """Validate an array, returning the validated array"""
        return validate_array(value, *self.__args__)

    def __instancecheck__(self, instance: Any) -> bool:
        """Whether ``instance`` is an array of any supported type that validates"""
        shape, dtype = self.__args__
        try:
            interface_cls = Interface.match(instance, fast=True)
            interface_cls(shape, dtype).validate(instance)
        except InterfaceError:
            return False
        return True

    def __getitem__(self, item: Any) -> NDArrayAlias:
        # Filling in type variables, like in list[NDArray[Shape[N]]][L[3]].
        # The base class would give a plain GenericAlias without normalizing.
        return self.__origin__[super().__getitem__(item).__args__]


class NDArray:
    """
    Constrained array type with :class:`.Shape` and dtype validation
    and serialization.

    ``NDArray[shape, dtype]`` is a generic alias of this class,
    and its arguments can be read with :func:`typing.get_args`.
    pydantic validates it with the relevant :ref:`interface <Interfaces>`,
    so any supported array type with a matching shape and dtype is valid.

    Calling ``NDArray`` or ``NDArray[shape, dtype]`` validates an array,
    and returns the validated array rather than an instance of this class.

    References:
        - https://docs.pydantic.dev/latest/usage/types/custom/#handling-third-party-types
    """

    def __class_getitem__(cls, args: Any) -> NDArrayAlias:
        if not isinstance(args, tuple) or (isinstance(args, tuple) and len(args) == 1):
            # just shape passed
            shape = args if not isinstance(args, TypeVar) else Any
            dtype = Any
        else:
            shape = args[0] if not isinstance(args[0], TypeVar) else Any
            dtype = args[1] if not isinstance(args[1], TypeVar) else Any
        return NDArrayAlias(cls, (_get_shape(shape), _get_dtype(dtype)))

    def __new__(cls, value: Any) -> NDArrayType:  # type: ignore[misc]
        """Validate an array with any shape and dtype"""
        return validate_array(value, Any, Any)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        shape, dtype = get_args(source) if get_origin(source) is cls else (Any, Any)
        return make_core_schema(shape, dtype, handler)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> core_schema.JsonSchema:
        return make_ndarray_json_schema(schema, handler)
