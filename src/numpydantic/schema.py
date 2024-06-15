"""
Helper functions for use with :class:`~numpydantic.NDArray` - see the note in
:mod:`~numpydantic.ndarray` for why these are separated.
"""

import hashlib
import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import nptyping.structure
import numpy as np
from pydantic import SerializationInfo
from pydantic_core import CoreSchema, core_schema
from pydantic_core.core_schema import ListSchema, ValidationInfo

from numpydantic import dtype as dt
from numpydantic.interface import Interface
from numpydantic.maps import np_to_python
from numpydantic.types import DtypeType, NDArrayType, ShapeType

if TYPE_CHECKING:
    from numpydantic import Shape

_handler_type = Callable[[Any], core_schema.CoreSchema]
_UNSUPPORTED_TYPES = (complex,)


def _numeric_dtype(dtype: DtypeType, _handler: _handler_type) -> CoreSchema:
    """Make a numeric dtype that respects min/max values from extended numpy types"""
    if dtype in (np.number,):
        dtype = float

    if issubclass(dtype, np.floating):
        info = np.finfo(dtype)
        schema = core_schema.float_schema(le=float(info.max), ge=float(info.min))
    elif issubclass(dtype, np.integer):
        info = np.iinfo(dtype)
        schema = core_schema.int_schema(le=int(info.max), ge=int(info.min))

    else:
        schema = _handler.generate_schema(dtype)

    return schema


def _lol_dtype(dtype: DtypeType, _handler: _handler_type) -> CoreSchema:
    """Get the innermost dtype schema to use in the generated pydantic schema"""
    if isinstance(dtype, nptyping.structure.StructureMeta):  # pragma: no cover
        raise NotImplementedError("Structured dtypes are currently unsupported")

    if isinstance(dtype, tuple):
        # if it's a meta-type that refers to a generic float/int, just make that
        if dtype == dt.Float:
            array_type = core_schema.float_schema()
        elif dtype == dt.Integer:
            array_type = core_schema.int_schema()
        elif dtype == dt.Complex:
            array_type = core_schema.any_schema()
        else:
            # make a union of dtypes recursively
            types_ = list(set(dtype))
            array_type = core_schema.union_schema(
                [_lol_dtype(t, _handler) for t in types_]
            )

    else:
        try:
            python_type = np_to_python[dtype]
        except KeyError as e:  # pragma: no cover
            # this should pretty much only happen in downstream/3rd-party interfaces
            # that use interface-specific types. those need to provide mappings back
            # to base python types (making this more streamlined is TODO)
            if dtype in np_to_python.values():
                # it's already a python type
                python_type = dtype
            else:
                raise ValueError(
                    "dtype given in model does not have a corresponding python base "
                    "type - add one to the `maps.np_to_python` dict"
                ) from e

        if python_type in _UNSUPPORTED_TYPES:
            array_type = core_schema.any_schema()
            # TODO: warn and log here
        elif python_type in (float, int):
            array_type = _numeric_dtype(dtype, _handler)
        else:
            array_type = _handler.generate_schema(python_type)

    return array_type


def list_of_lists_schema(shape: "Shape", array_type: CoreSchema) -> ListSchema:
    """
    Make a pydantic JSON schema for an array as a list of lists.

    For each item in the shape, create a list schema. In the innermost schema
    insert the passed ``array_type`` schema.

    This function is typically called from :func:`.make_json_schema`

    Args:
        shape (:class:`~numpydantic.Shape`): Shape determines the depth and max/min
            elements for each layer of list schema
        array_type ( :class:`pydantic_core.CoreSchema` ): The pre-rendered pydantic
            core schema to use in the innermost list entry
    """
    from numpydantic.shape import _is_range

    shape_parts = [part.strip() for part in shape.__args__[0].split(",")]
    # labels, if present
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
        inner_schema = array_type if list_schema is None else list_schema

        # make a label annotation, if we have one
        metadata = {"name": label} if label is not None else None

        # make the current level list schema, accounting for shape
        if arg == "*":
            list_schema = core_schema.list_schema(inner_schema, metadata=metadata)
        elif arg == "...":
            list_schema = _unbounded_shape(inner_schema, metadata=metadata)
        else:
            if _is_range(arg):
                arg_min, arg_max = arg.split("-")
                arg_min = None if arg_min == "*" else int(arg_min)
                arg_max = None if arg_max == "*" else int(arg_max)

            else:
                try:
                    arg = int(arg)
                    arg_min = arg
                    arg_max = arg
                except ValueError as e:

                    raise ValueError(
                        "Array shapes must be integers, wildcards, ellipses, or "
                        "ranges. Shape variables (for declaring that one dimension "
                        "must be the same size as another) are not supported because "
                        "it is impossible to express dynamic minItems/maxItems in "
                        "JSON Schema. "
                        "See: https://github.com/orgs/json-schema-org/discussions/730"
                    ) from e
            list_schema = core_schema.list_schema(
                inner_schema, min_length=arg_min, max_length=arg_max, metadata=metadata
            )
    return list_schema


def _hash_schema(schema: CoreSchema) -> str:
    """
    Make a hex-encoded 8-byte blake2b hash from a pydantic core schema.
    Collisions are really not important or likely here, but we do want the same schema
    to produce the same hash.
    """
    schema_str = json.dumps(
        schema, sort_keys=True, indent=None, separators=(",", ":")
    ).encode("utf-8")
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(schema_str)
    return hasher.hexdigest()


def _unbounded_shape(
    inner_type: CoreSchema, metadata: Optional[dict] = None
) -> core_schema.DefinitionsSchema:
    """
    Make a recursive schema that refers to itself using a hashed version of the inner
    type
    """

    schema_hash = _hash_schema(inner_type)
    array_ref = f"any-shape-array-{schema_hash}"

    schema = core_schema.definitions_schema(
        core_schema.list_schema(
            core_schema.definition_reference_schema(array_ref), metadata=metadata
        ),
        [
            core_schema.union_schema(
                [
                    core_schema.list_schema(
                        core_schema.definition_reference_schema(array_ref)
                    ),
                    inner_type,
                ],
                ref=array_ref,
            )
        ],
    )
    return schema


def make_json_schema(
    shape: ShapeType, dtype: DtypeType, _handler: _handler_type
) -> ListSchema:
    """
    Make a list of list JSON schema from a shape and a dtype.

    First resolves the dtype into a pydantic ``CoreSchema`` ,
    and then uses that with :func:`.list_of_lists_schema` .

    Args:
        shape ( ShapeType ): Specification of a shape, as a tuple or
            an nptyping ``Shape``
        dtype ( DtypeType ): A builtin type or numpy dtype
        _handler: The pydantic schema generation handler (see pydantic docs)

    Returns:
        :class:`pydantic_core.core_schema.ListSchema`
    """
    dtype_schema = _lol_dtype(dtype, _handler)

    # get the names of the shape constraints, if any
    if shape is Any:
        list_schema = _unbounded_shape(dtype_schema)
        # list_schema = core_schema.list_schema(core_schema.any_schema())
    else:
        list_schema = list_of_lists_schema(shape, dtype_schema)

    return list_schema


def get_validate_interface(shape: ShapeType, dtype: DtypeType) -> Callable:
    """
    Validate using a matching :class:`.Interface` class using its
    :meth:`.Interface.validate` method
    """

    def validate_interface(
        value: Any, info: Optional["ValidationInfo"] = None
    ) -> NDArrayType:
        interface_cls = Interface.match(value)
        interface = interface_cls(shape, dtype)
        value = interface.validate(value)
        return value

    return validate_interface


def _jsonize_array(value: Any, info: SerializationInfo) -> Union[list, dict]:
    """Use an interface class to render an array as JSON"""
    interface_cls = Interface.match_output(value)
    return interface_cls.to_json(value, info)
