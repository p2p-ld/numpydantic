"""
An array type for pydantic, with shape and dtype validation and JSON-Schema
serialization

.. note::

    This module should *only* have the :class:`.NDArray` class in it, because the
    type stub ``ndarray.pyi`` is only created for :class:`.NDArray` . Otherwise,
    type checkers will complain about using any helper functions elsewhere -
    those all belong in :mod:`numpydantic.schema` .

    NDArrayMeta is in this module even though it's excluded from the type stub.

"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    _ProtocolMeta,
    get_args,
    get_origin,
    runtime_checkable,
)

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

if TYPE_CHECKING:  # pragma: no cover
    from pydantic._internal._schema_generation_shared import (
        CallbackGetCoreSchemaHandler,
    )

    from numpydantic import Shape


def _get_shape(dtype_candidate: Any) -> Shape:
    """
    Normalize the shape argument of ``NDArray[shape, dtype]`` with :class:`.Shape`
    """
    from numpydantic.shape import Shape

    if dtype_candidate is Any or dtype_candidate is Shape:
        shape = Any
    elif get_origin(dtype_candidate) is Literal:
        shape_expression = dtype_candidate.__args__[0]
        shape = Shape[shape_expression]
    elif get_origin(dtype_candidate) is tuple:
        shape = Shape[get_args(dtype_candidate)]
    elif isinstance(dtype_candidate, (str, tuple)):
        shape = Shape[dtype_candidate]
    else:
        raise InvalidArgumentsError(
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


TShape = TypeVar("TShape")
TDType = TypeVar("TDType")


class NDArrayMeta(_ProtocolMeta):
    """
    Metaclass to provide class-level methods to NDArray protocol
    without suggesting they are part of the protocol definition.
    """

    __args__: tuple[ShapeType, DtypeType] = (Any, Any)

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

    def _dtype_to_str(cls, dtype: Any) -> str:
        if dtype is Any:
            result = "Any"
        elif isinstance(dtype, tuple):
            result = ", ".join([str(dt) for dt in dtype])
        else:
            result = str(dtype)
        return result

    def __getitem__(cls, args: type[Any] | tuple[type[Any], type[Any]]):
        if not isinstance(args, tuple) or (isinstance(args, tuple) and len(args) == 1):
            # just shape passed
            shape = args if not isinstance(args, TypeVar) else Any
            dtype = Any
        else:
            shape = args[0] if not isinstance(args[0], TypeVar) else Any
            dtype = args[1] if not isinstance(args[0], TypeVar) else Any

        shape = _get_shape(shape)
        dtype = _get_dtype(dtype)

        return type(cls.__name__, (cls,), {**cls.__dict__, "__args__": (shape, dtype)})

    def __get_pydantic_core_schema__(
        cls,
        _source_type: NDArray | Any,
        _handler: CallbackGetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        shape, dtype = cls.__args__
        shape: ShapeType
        dtype: DtypeType

        serialization = core_schema.plain_serializer_function_ser_schema(
            jsonize_array, when_used="json", info_arg=True
        )

        # make core schema for json schema, store it and any model definitions
        # so that we can use them when rendering json schema
        json_schema = make_json_schema(shape, dtype, _handler)

        if (
            not hasattr(_source_type, "__class__")
            or _source_type.__class__ is not NDArrayMeta
        ):
            if hasattr(_source_type, "proxy_for"):
                interface: type[Interface] = _source_type.proxy_for()
                isinstance_schema = core_schema.union_schema(
                    [
                        core_schema.is_instance_schema(itype)
                        for itype in list(interface.input_types) + [_source_type]
                    ]
                )
            else:
                isinstance_schema = core_schema.is_instance_schema(_source_type)

            return core_schema.chain_schema(
                [
                    isinstance_schema,
                    core_schema.with_info_plain_validator_function(
                        get_validate_interface(shape, dtype)
                    ),
                ],
                serialization=serialization,
                metadata=json_schema,
            )
        else:
            return core_schema.with_info_plain_validator_function(
                get_validate_interface(shape, dtype),
                serialization=serialization,
                metadata=json_schema,
            )

    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> core_schema.JsonSchema:
        shape, dtype = cls.__args__
        json_schema = handler(schema["metadata"])
        json_schema = handler.resolve_ref_schema(json_schema)

        if (
            not isinstance(dtype, tuple)
            and dtype.__module__
            not in (
                "builtins",
                "typing",
                "types",
            )
            and hasattr(dtype, "__name__")
        ):
            json_schema["dtype"] = ".".join([dtype.__module__, dtype.__name__])

        return json_schema


@runtime_checkable
class NDArray(Protocol[TShape, TDType], metaclass=NDArrayMeta):
    """
    Constrained array type with :class:`.Shape` and dtype validation
    and serialization.

    This class is not intended to be instantiable, and support for static type
    checking is limited,
    it implements the ``__get_pydantic_core_schema__`` method to invoke
    the relevant :ref:`interface <Interfaces>` for validation and serialization.

    It is callable, however, which validates and attempts to coerce input to a
    supported array type.
    There is no such thing as an "NDArray instance," but one can think of it
    as a validating passthrough callable.

    References:
        - https://docs.pydantic.dev/latest/usage/types/custom/#handling-third-party-types
    """

    shape = np.ndarray.shape
    dtype = np.ndarray.dtype

    def __getitem__(self: Any, key: Any) -> Any: ...

    def __setitem__(self: Any, key: Any, value: Any) -> Any: ...
