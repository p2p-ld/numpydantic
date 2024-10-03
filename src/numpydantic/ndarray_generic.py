from typing import Protocol, TypeVar, runtime_checkable

from typing_extensions import Unpack

from numpydantic.types import DtypeType

# Shape = TypeVarTuple("Shape")
# Shape = tuple[int, ...]
Shape = TypeVar("Shape", bound=tuple[int, ...])
DType = TypeVar("DType", bound=DtypeType)


@runtime_checkable
class NDArray(Protocol[Shape, DType]):
    """v2 generic protocol ndarray"""

    @property
    def dtype(self) -> DType:
        """dtype"""

    @property
    def shape(self) -> Unpack[Shape]:
        """shape"""


#
#
# def __get_pydantic_core_schema__(
#     typ: Type, handler: CallbackGetCoreSchemaHandler
# ) -> core_schema.CoreSchema:
#     args = get_args(typ)
#     if len(args) == 0:
#         shape, dtype = Any, Any
#     elif len(args) == 1:
#         shape, dtype = args[0], Any
#     elif len(args) == 2:
#         shape, dtype = args[0], args[1]
#     else:
#         shape, dtype = args[:-1], args[-1]
#
#     json_schema = make_json_schema(shape, dtype, handler)
#     return core_schema.with_info_plain_validator_function(
#         get_validate_interface(shape, dtype),
#         serialization=core_schema.plain_serializer_function_ser_schema(
#             jsonize_array, when_used="json", info_arg=True
#         ),
#         metadata=json_schema,
#     )


#
# def __get_pydantic_json_schema__(
#     schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
# ) -> core_schema.JsonSchema:
#     # shape, dtype = cls.__args__
#     json_schema = handler(schema["metadata"])
#     json_schema = handler.resolve_ref_schema(json_schema)
#
#     # if not isinstance(dtype, tuple) and dtype.__module__ not in (
#     #     "builtins",
#     #     "typing",
#     # ):
#     #     json_schema["dtype"] = ".".join([dtype.__module__, dtype.__name__])
#
#     return json_schema


# NDArray = Annotated[
#     _NDArray[Unpack[Shape], DType],
#     GetPydanticSchema(__get_pydantic_core_schema__),
#     # GetJsonSchemaFunction(__get_pydantic_json_schema__),
# ]
