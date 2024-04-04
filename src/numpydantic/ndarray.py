"""
Extension of nptyping NDArray for pydantic that allows for JSON-Schema serialization

* Order to store data in (row first)
"""

import base64
import sys
from collections.abc import Callable
from copy import copy
from typing import Any, Tuple, TypeVar, cast, Union

import blosc2
import nptyping.structure
import numpy as np
from nptyping import Shape
from nptyping.ndarray import NDArrayMeta as _NDArrayMeta
from nptyping.nptyping_type import NPTypingType
from nptyping.shape_expression import check_shape
from pydantic_core import core_schema
from pydantic_core.core_schema import ListSchema

from numpydantic.interface import Interface
from numpydantic.maps import np_to_python

# from numpydantic.proxy import NDArrayProxy
from numpydantic.types import DtypeType, NDArrayType, ShapeType

COMPRESSION_THRESHOLD = 16 * 1024
"""
Arrays larger than this size (in bytes) will be compressed and b64 encoded when 
serializing to JSON.
"""


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
    for arg, label in zip(shape_args, shape_labels, strict=False):
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


def jsonize_array(array: NDArrayType) -> list | dict:
    """
    Render an array to base python types that can be serialized to JSON

    For small arrays, returns a list of lists.

    If the array is over :class:`.COMPRESSION_THRESHOLD` bytes, use :func:`.compress_array`
    to return a compressed b64 encoded string.

    Args:
        array (:class:`np.ndarray`, :class:`dask.DaskArray`): Array to render as a list!
    """
    # if isinstance(array, DaskArray):
    #    arr = array.__array__()
    # elif isinstance(array, NDArrayProxy):
    #    arr = array[:]
    # else:
    #    arr = array
    arr = array

    # If we're larger than 16kB then compress array!
    if sys.getsizeof(arr) > COMPRESSION_THRESHOLD:
        packed = blosc2.pack_array2(arr)
        packed = base64.b64encode(packed)
        ret = {
            "array": packed,
            "shape": copy(arr.shape),
            "dtype": copy(arr.dtype.name),
            "unpack_fns": ["base64.b64decode", "blosc2.unpack_array2"],
        }
        return ret
    else:
        return arr.tolist()


def get_validate_shape(shape: Shape) -> Callable:
    """
    Get a closure around a shape validation function that includes the shape definition
    """

    def validate_shape(value: Any) -> np.ndarray:
        assert shape is Any or check_shape(
            value.shape, shape
        ), f"Invalid shape! expected shape {shape.prepared_args}, got shape {value.shape}"

        return value

    return validate_shape


def get_validate_interface(shape: ShapeType, dtype: DtypeType) -> Callable:
    """
    Validate using a matching :class:`.Interface` class using its :meth:`.Interface.validate` method
    """

    def validate_interface(value: Any, info) -> NDArrayType:
        interface_cls = Interface.match(value)
        interface = interface_cls(shape, dtype)
        value = interface.validate(value)
        return value

    return validate_interface


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
    Kept here to allow for hooking into metaclass, which has
    been necessary on and off as we work this class into a stable
    state
    """


T = TypeVar("T")


class NDArray(NPTypingType, metaclass=NDArrayMeta):
    """
    Constrained array type allowing npytyping syntax for dtype and shape validation and serialization.

    Integrates with pydantic such that
    - JSON schema for list of list encoding
    - Serialized as LoL, with automatic compression for large arrays
    - Automatic coercion from lists on instantiation

    Also supports validation on :class:`.NDArrayProxy` types for lazy loading.

    References:
        - https://docs.pydantic.dev/latest/usage/types/custom/#handling-third-party-types
    """

    def __init__(self: T):
        pass

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
                        get_validate_interface(shape, dtype)
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                jsonize_array, when_used="json"
            ),
        )


NDArray = cast(Union[np.ndarray, list[int]], NDArray)
# NDArray = cast(Union[Interface.array_types()], NDArray)
