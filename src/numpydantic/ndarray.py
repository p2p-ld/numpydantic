"""
Extension of nptyping NDArray for pydantic that allows for JSON-Schema serialization

* Order to store data in (row first)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Tuple, Union

import nptyping.structure
import numpy as np
from nptyping import Shape
from nptyping.error import InvalidArgumentsError
from nptyping.ndarray import NDArrayMeta as _NDArrayMeta
from nptyping.nptyping_type import NPTypingType
from nptyping.structure import Structure
from nptyping.structure_expression import check_type_names
from nptyping.typing_ import (
    dtype_per_name,
)
from pydantic_core import core_schema
from pydantic_core.core_schema import ListSchema

from numpydantic.dtype import DType
from numpydantic.interface import Interface
from numpydantic.maps import np_to_python

# from numpydantic.proxy import NDArrayProxy
from numpydantic.types import DtypeType, NDArrayType, ShapeType

if TYPE_CHECKING:
    from pydantic import ValidationInfo


def list_of_lists_schema(shape: Shape, array_type_handler: dict) -> ListSchema:
    """Make a pydantic JSON schema for an array as a list of lists."""
    shape_parts = shape.__args__[0].split(",")
    split_parts = [
        p.split(" ")[1] if len(p.split(" ")) == 2 else None for p in shape_parts
    ]

    # Construct a list of list schema
    # go in reverse order - construct list schemas such that
    # the final schema is the one that checks the first dimension
    shape_labels = reversed(split_parts)
    shape_args = reversed(shape.prepared_args)
    list_schema = None
    for arg, label in zip(shape_args, shape_labels):
        # which handler to use? for the first we use the actual type
        # handler, everywhere else we use the prior list handler
        inner_schema = array_type_handler if list_schema is None else list_schema

        # make a label annotation, if we have one
        metadata = {"name": label} if label is not None else None

        # make the current level list schema, accounting for shape
        if arg == "*":
            list_schema = core_schema.list_schema(inner_schema, metadata=metadata)
        else:
            arg = int(arg)
            list_schema = core_schema.list_schema(
                inner_schema, min_length=arg, max_length=arg, metadata=metadata
            )
    return list_schema


def _get_validate_interface(shape: ShapeType, dtype: DtypeType) -> Callable:
    """
    Validate using a matching :class:`.Interface` class using its
    :meth:`.Interface.validate` method
    """

    def validate_interface(value: Any, info: "ValidationInfo") -> NDArrayType:
        interface_cls = Interface.match(value)
        interface = interface_cls(shape, dtype)
        value = interface.validate(value)
        return value

    return validate_interface


def _jsonize_array(value: Any) -> Union[list, dict]:
    interface_cls = Interface.match_output(value)
    return interface_cls.to_json(value)


def coerce_list(value: Any) -> np.ndarray:
    """
    If a value is passed as a list or list of lists, try and coerce it into an array
    rather than failing validation.
    """
    if isinstance(value, list):
        value = np.array(value)
    return value


class NDArrayMeta(_NDArrayMeta, implementation="NDArray"):
    """
    Hooking into nptyping's array metaclass to override methods pending
    completion of the transition away from nptyping
    """

    def _get_dtype(cls, dtype_candidate: Any) -> DType:
        """
        Override of base _get_dtype method to allow for compound tuple types
        """
        is_dtype = isinstance(dtype_candidate, type) and issubclass(
            dtype_candidate, np.generic
        )
        if dtype_candidate is Any:
            dtype = Any
        elif is_dtype:
            dtype = dtype_candidate
        elif issubclass(dtype_candidate, Structure):
            dtype = dtype_candidate
            check_type_names(dtype, dtype_per_name)
        elif cls._is_literal_like(dtype_candidate):
            structure_expression = dtype_candidate.__args__[0]
            dtype = Structure[structure_expression]
            check_type_names(dtype, dtype_per_name)
        elif isinstance(dtype_candidate, tuple):
            dtype = tuple([cls._get_dtype(dt) for dt in dtype_candidate])
        else:
            raise InvalidArgumentsError(
                f"Unexpected argument '{dtype_candidate}', expecting"
                " Structure[<StructureExpression>]"
                " or Literal[<StructureExpression>]"
                " or a dtype"
                " or typing.Any."
            )
        return dtype


class NDArray(NPTypingType, metaclass=NDArrayMeta):
    """
    Constrained array type allowing npytyping syntax for dtype and shape validation
    and serialization.

    Integrates with pydantic such that
    - JSON schema for list of list encoding
    - Serialized as LoL, with automatic compression for large arrays
    - Automatic coercion from lists on instantiation

    Also supports validation on :class:`.NDArrayProxy` types for lazy loading.

    References:
        - https://docs.pydantic.dev/latest/usage/types/custom/#handling-third-party-types
    """

    __args__: Tuple[ShapeType, DtypeType] = (Any, Any)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: "NDArray",
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        shape, dtype = _source_type.__args__
        shape: ShapeType
        dtype: DtypeType

        # get pydantic core schema for the given specified type
        if isinstance(dtype, nptyping.structure.StructureMeta):
            raise NotImplementedError("Finish handling structured dtypes!")
            # functools.reduce(operator.or_, [int, float, str])
        else:
            if isinstance(dtype, tuple):
                types_ = list(set([np_to_python[dt] for dt in dtype]))
                # TODO: better type filtering - explicitly model what
                # numeric types are supported by JSON schema
                types_ = [t for t in types_ if t not in (complex,)]
                schemas = [_handler.generate_schema(dt) for dt in types_]
                array_type_handler = core_schema.union_schema(schemas)

            else:
                array_type_handler = _handler.generate_schema(np_to_python[dtype])

        # get the names of the shape constraints, if any
        if shape is Any:
            list_schema = core_schema.list_schema(core_schema.any_schema())
        else:
            list_schema = list_of_lists_schema(shape, array_type_handler)

        return core_schema.json_or_python_schema(
            json_schema=list_schema,
            python_schema=core_schema.chain_schema(
                [
                    core_schema.no_info_plain_validator_function(coerce_list),
                    core_schema.with_info_plain_validator_function(
                        _get_validate_interface(shape, dtype)
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _jsonize_array, when_used="json"
            ),
        )
