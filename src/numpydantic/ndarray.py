"""
Extension of nptyping NDArray for pydantic that allows for JSON-Schema serialization

.. note::

    This module should *only* have the :class:`.NDArray` class in it, because the
    type stub ``ndarray.pyi`` is only created for :class:`.NDArray` . Otherwise,
    type checkers will complain about using any helper functions elsewhere -
    those all belong in :mod:`numpydantic.schema` .

    Keeping with nptyping's style, NDArrayMeta is in this module even if it's
    excluded from the type stub.

"""

from typing import TYPE_CHECKING, Any, Literal, Tuple, get_origin

import numpy as np
from pydantic import GetJsonSchemaHandler
from pydantic_core import core_schema

from numpydantic.dtype import DType
from numpydantic.exceptions import InterfaceError
from numpydantic.interface import Interface
from numpydantic.maps import python_to_nptyping
from numpydantic.schema import (
    get_validate_interface,
    make_json_schema,
)
from numpydantic.serialization import jsonize_array
from numpydantic.types import DtypeType, NDArrayType, ShapeType
from numpydantic.validation.dtype import is_union
from numpydantic.vendor.nptyping.error import InvalidArgumentsError
from numpydantic.vendor.nptyping.ndarray import NDArrayMeta as _NDArrayMeta
from numpydantic.vendor.nptyping.nptyping_type import NPTypingType
from numpydantic.vendor.nptyping.structure import Structure
from numpydantic.vendor.nptyping.structure_expression import check_type_names
from numpydantic.vendor.nptyping.typing_ import (
    dtype_per_name,
)

if TYPE_CHECKING:  # pragma: no cover
    from nptyping.base_meta_classes import SubscriptableMeta
    from pydantic._internal._schema_generation_shared import (
        CallbackGetCoreSchemaHandler,
    )

    from numpydantic import Shape


class NDArrayMeta(_NDArrayMeta, implementation="NDArray"):
    """
    Hooking into nptyping's array metaclass to override methods pending
    completion of the transition away from nptyping
    """

    if TYPE_CHECKING:  # pragma: no cover
        __getitem__ = SubscriptableMeta.__getitem__

    def __call__(cls, val: NDArrayType) -> NDArrayType:
        """Call ndarray as a validator function"""
        return get_validate_interface(cls.__args__[0], cls.__args__[1])(val)

    def __instancecheck__(self, instance: Any):
        """
        Extended type checking that determines whether

        1) the ``type`` of the given instance is one of those in
            :meth:`.Interface.input_types`

        but also

        2) it satisfies the constraints set on the :class:`.NDArray` annotation

        Args:
            instance (:class:`typing.Any`): Thing to check!

        Returns:
            bool: ``True`` if matches constraints, ``False`` otherwise.
        """
        shape, dtype = self.__args__
        try:
            interface_cls = Interface.match(instance, fast=True)
            interface = interface_cls(shape, dtype)
            _ = interface.validate(instance)
            return True
        except InterfaceError:
            return False

    def _is_literal_like(cls, item: Any) -> bool:
        """
        Changes from nptyping:
        - doesn't just ducktype for literal but actually, yno, checks for being literal
        """
        return get_origin(item) is Literal

    def _get_shape(cls, dtype_candidate: Any) -> "Shape":
        """
        Override of base method to use our local definition of shape
        """
        from numpydantic.validation.shape import Shape

        if dtype_candidate is Any or dtype_candidate is Shape:
            shape = Any
        elif issubclass(dtype_candidate, Shape):
            shape = dtype_candidate
        elif cls._is_literal_like(dtype_candidate):
            shape_expression = dtype_candidate.__args__[0]
            shape = Shape[shape_expression]
        else:
            raise InvalidArgumentsError(
                f"Unexpected argument '{dtype_candidate}', expecting"
                " Shape[<ShapeExpression>]"
                " or Literal[<ShapeExpression>]"
                " or typing.Any."
            )
        return shape

    def _get_dtype(cls, dtype_candidate: Any) -> DType:
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
        elif issubclass(dtype_candidate, Structure):  # pragma: no cover
            dtype = dtype_candidate
            check_type_names(dtype, dtype_per_name)
        elif cls._is_literal_like(dtype_candidate):  # pragma: no cover
            structure_expression = dtype_candidate.__args__[0]
            dtype = Structure[structure_expression]
            check_type_names(dtype, dtype_per_name)
        elif isinstance(dtype_candidate, tuple):  # pragma: no cover
            dtype = tuple([cls._get_dtype(dt) for dt in dtype_candidate])
        else:
            # arbitrary dtype - allow failure elsewhere :)
            dtype = dtype_candidate

        return dtype

    def _dtype_to_str(cls, dtype: Any) -> str:
        if dtype is Any:
            result = "Any"
        elif issubclass(dtype, Structure):
            result = str(dtype)
        elif isinstance(dtype, tuple):
            result = ", ".join([str(dt) for dt in dtype])
        else:
            result = str(dtype)
        return result


class NDArray(NPTypingType, metaclass=NDArrayMeta):
    """
    Constrained array type allowing npytyping syntax for dtype and shape validation
    and serialization.

    This class is not intended to be instantiated or used for type checking, it
    implements the ``__get_pydantic_core_schema__` method to invoke
    the relevant :ref:`interface <Interfaces>` for validation and serialization.

    References:
        - https://docs.pydantic.dev/latest/usage/types/custom/#handling-third-party-types
    """

    __args__: Tuple[ShapeType, DtypeType] = (Any, Any)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: "NDArray",
        _handler: "CallbackGetCoreSchemaHandler",
    ) -> core_schema.CoreSchema:
        shape, dtype = _source_type.__args__
        shape: ShapeType
        dtype: DtypeType

        # make core schema for json schema, store it and any model definitions
        # note that there is a big of fragility in this function,
        # as we need to access a private method of _handler to
        # flatten out the json schema. See help(make_json_schema)
        json_schema = make_json_schema(shape, dtype, _handler)

        return core_schema.with_info_plain_validator_function(
            get_validate_interface(shape, dtype),
            serialization=core_schema.plain_serializer_function_ser_schema(
                jsonize_array, when_used="json", info_arg=True
            ),
            metadata=json_schema,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> core_schema.JsonSchema:
        shape, dtype = cls.__args__
        json_schema = handler(schema["metadata"])
        json_schema = handler.resolve_ref_schema(json_schema)

        if not isinstance(dtype, tuple) and dtype.__module__ not in (
            "builtins",
            "typing",
        ):
            json_schema["dtype"] = ".".join([dtype.__module__, dtype.__name__])

        return json_schema
