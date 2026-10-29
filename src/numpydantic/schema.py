"""
Helper functions for use with :class:`~numpydantic.NDArray` - see the note in
:mod:`~numpydantic.ndarray` for why these are separated.
"""

import dataclasses
import hashlib
import json
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    get_args,
    get_origin,
)

import numpy as np
from annotated_types import Interval
from pydantic import BaseModel, GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema
from pydantic_core.core_schema import ListSchema

from numpydantic import dtype as dt
from numpydantic.interface import Interface
from numpydantic.maps import np_to_python
from numpydantic.serialization import jsonize_array
from numpydantic.shape._variadic import _split_variadic
from numpydantic.types import DtypeType, ShapeType
from numpydantic.validation.array import get_validate_interface
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
    if tail is not None:
        inner_schema = _recursive_shape(array_type, _list_schema_kwargs(tail))
    else:
        inner_schema = array_type

    for arg in reversed(fixed):
        # which handler to use? for the first we use the actual type
        # handler, everywhere else we use the prior list handler
        inner_schema = core_schema.list_schema(inner_schema, **_list_schema_kwargs(arg))

    return inner_schema


def _list_schema_kwargs(arg: Any) -> dict:
    kwargs = {}
    if get_origin(arg) is Annotated:
        typ, *annotations = get_args(arg)
    else:
        typ, annotations = arg, []

    labels = [item for item in annotations if isinstance(item, str)]
    if labels:
        kwargs["metadata"] = {"name": labels[0]}

    if get_origin(typ) is Literal:
        # TODO: support multiple exact lengths as unions
        kwargs["min_length"] = kwargs["max_length"] = get_args(typ)[0]

    elif interval := next(
        (annotation for annotation in annotations if isinstance(annotation, Interval)),
        None,
    ):
        if interval.ge is not None:
            kwargs["min_length"] = interval.ge
        elif interval.gt is not None:
            kwargs["min_length"] = interval.gt + 1

        if interval.le is not None:
            kwargs["max_length"] = interval.le
        elif interval.lt is not None:
            kwargs["max_length"] = interval.lt - 1

    return kwargs


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
        shape ( ShapeType ): A normalized shape, as returned by ``Shape[...]``:
            a ``tuple[...]``, or ``Any``
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


def make_core_schema(
    shape: ShapeType,
    dtype: DtypeType,
    handler: "CallbackGetCoreSchemaHandler",
    source_type: Any = None,
) -> CoreSchema:
    """
    The core schema for an array with a normalized shape and dtype.

    With a ``source_type``, as with :func:`.NDArraySchema`, the value also has to be
    an instance of that type, or of an input type of the interface it's a proxy for.
    """
    serialization = core_schema.plain_serializer_function_ser_schema(
        jsonize_array, when_used="json", info_arg=True
    )
    # The JSON schema's core schema is made here, where the dtype's schema
    # can be generated, and rendered by make_ndarray_json_schema.
    metadata = {
        "numpydantic": {
            "json_schema": make_json_schema(shape, dtype, handler),
            "dtype": dtype,
        }
    }
    validate_interface = get_validate_interface(
        shape,
        dtype,
        cross_field_validation=_supports_cross_field_validation(handler),
    )

    if source_type is None:
        return core_schema.with_info_plain_validator_function(
            validate_interface, serialization=serialization, metadata=metadata
        )

    if hasattr(source_type, "proxy_for"):
        interface: type[Interface] = source_type.proxy_for()
        isinstance_schema = core_schema.union_schema(
            [
                core_schema.is_instance_schema(itype)
                for itype in list(interface.input_types) + [source_type]
            ]
        )
    else:
        isinstance_schema = core_schema.is_instance_schema(source_type)
    return core_schema.chain_schema(
        [
            isinstance_schema,
            core_schema.with_info_plain_validator_function(validate_interface),
        ],
        serialization=serialization,
        metadata=metadata,
    )


def make_ndarray_json_schema(
    schema: CoreSchema, handler: GetJsonSchemaHandler
) -> dict[str, Any]:
    """Render the JSON schema stored by :func:`.make_core_schema`"""
    stored = schema["metadata"]["numpydantic"]
    dtype = stored["dtype"]
    json_schema = handler.resolve_ref_schema(handler(stored["json_schema"]))

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


def _supports_cross_field_validation(handler: "CallbackGetCoreSchemaHandler") -> bool:
    """
    Whether shape variable sizes can be shared between fields through ``info.data``.

    ``info.data`` becomes the ``__dict__`` of a model or dataclass,
    but for a TypedDict it's the dict that's returned, so the sizes would show up in it.
    """
    # pydantic doesn't tell a field what it belongs to.
    # If this private attribute changes, sizes are only matched within one array.
    generator = getattr(handler, "_generate_schema", None)
    stack = getattr(generator, "model_type_stack", None)
    owner = stack.get() if stack is not None else None
    return isinstance(owner, type) and (
        issubclass(owner, BaseModel) or dataclasses.is_dataclass(owner)
    )
