"""
Helper functions for use with :class:`~numpydantic.NDArray` - see the note in
:mod:`~numpydantic.ndarray` for why these are separated.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    get_args,
    get_origin,
)

import numpy as np
from annotated_types import Interval
from pydantic import BaseModel
from pydantic_core import CoreSchema, core_schema
from pydantic_core.core_schema import ListSchema, ValidationInfo

from numpydantic import dtype as dt
from numpydantic.exceptions import DtypeError, ShapeError
from numpydantic.interface import Interface
from numpydantic.maps import np_to_python
from numpydantic.shape._variadic import _split_variadic
from numpydantic.types import DtypeType, NDArrayType, ShapeType
from numpydantic.validation.dtype import is_union

if TYPE_CHECKING:  # pragma: no cover
    from pydantic._internal._schema_generation_shared import (
        CallbackGetCoreSchemaHandler,
    )

    from numpydantic import Shape


def _numeric_dtype(
    dtype: DtypeType, _handler: "CallbackGetCoreSchemaHandler"
) -> CoreSchema:
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


def _lol_dtype(
    dtype: DtypeType, _handler: "CallbackGetCoreSchemaHandler"
) -> CoreSchema:
    """Get the innermost dtype schema to use in the generated pydantic schema"""
    if isinstance(dtype, tuple):
        # if it's a meta-type that refers to a generic float/int, just make that
        if dtype in (dt.Float, dt.Number):
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
    elif is_union(dtype):
        array_type = core_schema.union_schema(
            [_lol_dtype(t, _handler) for t in get_args(dtype)]
        )
    else:
        try:
            python_type = np_to_python[dtype]
        except KeyError:  # pragma: no cover
            # this should pretty much only happen in downstream/3rd-party interfaces
            # that use interface-specific types. those need to provide mappings back
            # to base python types (making this more streamlined is TODO)
            if dtype in np_to_python.values():
                # it's already a python type
                python_type = dtype
            elif issubclass(dtype, BaseModel):
                python_type = dtype
            else:
                # does this need a warning?
                python_type = Any

        if python_type in (float, int):
            array_type = _numeric_dtype(dtype, _handler)
        elif python_type is bool:
            array_type = core_schema.bool_schema()
        elif python_type is Any:
            array_type = core_schema.any_schema()
        elif issubclass(python_type, BaseModel):
            # We a) need to materialize the model schema rather than reference it
            # since there isn't a good way to propagate the reference through the
            # handler,
            # and b) can't use the handler to generate the schema
            # since it can only generate the reference
            # caching it for full materialization later.
            # the easiest way to get the model's schema is just to ... get it,
            # which may be a bit expensive with the copy.
            # see https://github.com/p2p-ld/numpydantic/issues/48
            # if this proves to be prohibitively expensive, we can find a better way.
            array_type = python_type.__pydantic_core_schema__.copy()
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

    # Construct a list of list schema
    # go in reverse order - construct list schemas such that
    # the final schema is the one that checks the first dimension
    shape_args = get_args(shape)
    fixed, tail = _split_variadic(shape_args)

    # if we have a tail, that's the inner schema
    if tail:
        inner_schema = _recursive_shape(array_type, _list_schema_kwargs(tail))
    else:
        inner_schema = array_type

    for arg in reversed(fixed):
        # which handler to use? for the first we use the actual type
        # handler, everywhere else we use the prior list handler
        inner_schema = core_schema.list_schema(inner_schema, **_list_schema_kwargs(arg))

    return inner_schema


def _list_schema_kwargs(arg: Any) -> dict:
    # make a label annotation, if we have one
    # metadata = {"name": label} if label is not None else None
    kwargs = {}
    # unwrap annotations from inner value
    if get_origin(arg) is Annotated:
        args = get_args(arg)
        typ = args[0]
        annotations = args[1:]
    else:
        typ = arg
        annotations = tuple()

    labels = [item for item in annotations if isinstance(item, str)]
    if labels:
        kwargs["metadata"] = {"name": labels[0]}

    # make the current level list schema, accounting for shape
    try:
        if typ is int:
            return kwargs
        elif get_origin(typ) is Literal:
            # TODO: support multiple exact lengths
            val = int(get_args(typ)[0])
            kwargs["min_length"] = val
            kwargs["max_length"] = val

        elif any(isinstance(annotation, Interval) for annotation in annotations):
            interval = next(
                annotation
                for annotation in annotations
                if isinstance(annotation, Interval)
            )
            if interval.ge:
                kwargs["min_length"] = interval.ge
            elif interval.gt:
                kwargs["min_length"] = interval.gt - 1

            if interval.le:
                kwargs["max_length"] = interval.le
            elif interval.lt:
                kwargs["max_length"] = interval.lt + 1

        return kwargs
    except ValueError as e:  # pragma: no cover
        raise ValueError(
            "Array shapes must be integers, wildcards, ellipses, or "
            "ranges. Shape variables (for declaring that one dimension "
            "must be the same size as another) are not supported because "
            "it is impossible to express dynamic minItems/maxItems in "
            "JSON Schema. "
            "See: https://github.com/orgs/json-schema-org/discussions/730 "
            f"Instead, got {arg}"
        ) from e


def _hash_schema(schema: CoreSchema, list_kwargs: dict | None = None) -> str:
    """
    Make a hex-encoded 8-byte blake2b hash from a pydantic core schema.
    Collisions are really not important or likely here, but we do want the same schema
    to produce the same hash.
    """
    schema_str = json.dumps(
        schema,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
        default=lambda x: None,
    ).encode("utf-8")
    if list_kwargs:
        schema_str += json.dumps(
            list_kwargs,
            sort_keys=True,
            indent=None,
            separators=(",", ":"),
            default=lambda x: None,
        ).encode("utf-8")
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(schema_str)
    return hasher.hexdigest()


def _recursive_shape(
    inner_type: CoreSchema, list_kwargs: dict | None = None
) -> core_schema.DefinitionsSchema:
    """
    Make a recursive schema that refers to itself using a hashed version of the inner
    type
    """
    schema_hash = _hash_schema(inner_type, list_kwargs)
    if list_kwargs is None:
        list_kwargs = {}

    array_ref = f"any-shape-array-{schema_hash}"

    schema = core_schema.definitions_schema(
        core_schema.list_schema(
            core_schema.definition_reference_schema(array_ref), **list_kwargs
        ),
        [
            core_schema.union_schema(
                [
                    core_schema.list_schema(
                        core_schema.definition_reference_schema(array_ref),
                        **list_kwargs,
                    ),
                    inner_type,
                ],
                ref=array_ref,
            )
        ],
    )
    return schema


def make_json_schema(
    shape: ShapeType, dtype: DtypeType, _handler: "CallbackGetCoreSchemaHandler"
) -> ListSchema:
    """
    Make a list of list pydantic core schema for an array from a shape and a dtype.
    Used to generate JSON schema in the containing model, but not for validation,
    which is handled by interfaces.

    First resolves the dtype into a pydantic ``CoreSchema`` ,
    and then uses that with :func:`.list_of_lists_schema` .

    Args:
        shape ( ShapeType ): Specification of a shape, as a tuple or
            an nptyping ``Shape``
        dtype ( DtypeType ): A builtin type or numpy dtype
        _handler: The pydantic schema generation handler (see pydantic docs).
            Needed to generate the schema for the dtype
            if we don't know how to do it ourselves.

    Returns:
        :class:`pydantic_core.core_schema.ListSchema`
    """
    dtype_schema = _lol_dtype(dtype, _handler)

    # get the names of the shape constraints, if any
    if shape is Any:
        list_schema = _recursive_shape(dtype_schema)
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
        try:
            value = interface.validate(value)
        except (ShapeError, DtypeError) as e:
            _warn_about_possible_missing_deps(e, value)
        return value

    return validate_interface


def _warn_about_possible_missing_deps(e: Exception, value: Any) -> None:
    if isinstance(value, (str, Path)):
        new_ex = type(e)(
            str(e) + "\nThe passed input looks like a path, "
            "make sure you have the required dependencies installed to load the array!"
        )
        raise new_ex
    else:
        raise e
