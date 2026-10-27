"""
Interface to zarr arrays
"""

import contextlib
import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    GetJsonSchemaHandler,
    GetPydanticSchema,
    PydanticUserError,
    SerializationInfo,
    TypeAdapter,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from numpydantic.interface.interface import Interface, JsonDict
from numpydantic.interface.typing import ConstructorSpec, InterfaceTyping

try:
    import zarr
    from zarr import Array as ZarrArray
    from zarr.buffer import default_buffer_prototype
    from zarr.core.metadata import (
        ArrayMetadata,
        ArrayMetadataDict,
        ArrayV2Metadata,
        ArrayV3Metadata,
    )
    from zarr.core.sync import sync
    from zarr.storage import LocalStore, ZipStore
except ImportError:  # pragma: no cover
    ZarrArray = None
    ArrayMetadata = Any
    ArrayMetadataDict = None
    ArrayV2Metadata = None
    ArrayV3Metadata = None
    LocalStore = None
    ZipStore = None


@dataclass
class ZarrArrayPath:
    """
    Map to an array within a zarr store.

    See :func:`zarr.open`
    """

    file: Path | str
    """Location of Zarr store file or directory"""
    path: str | None = None
    """Path to array within hierarchical zarr store"""

    def open(self, mode: str = "a", **kwargs: dict) -> ZarrArray:
        """Open the zarr array at the provided path"""
        store = str(self.file)
        # zarr 2 opened .zip paths as zip stores automatically, zarr 3 does not
        if store.endswith(".zip"):
            store = ZipStore(store, mode=mode if mode in ("r", "w") else "a")
        return zarr.open(store, path=self.path, mode=mode, **kwargs)

    @classmethod
    def from_iterable(cls, spec: Sequence) -> "ZarrArrayPath":
        """
        Construct a :class:`.ZarrArrayPath` specifier from an iterable,
        rather than kwargs
        """
        if len(spec) == 1:
            return ZarrArrayPath(file=spec[0])
        elif len(spec) == 2:
            return ZarrArrayPath(file=spec[0], path=spec[1])
        else:
            raise ValueError("Only len 1-2 iterables can be used for a ZarrArrayPath")


def hexdigest(array: ZarrArray) -> str:
    """
    SHA256 of the raw bytes of the files that make up a zarr array,
    read in sorted path order.

    .. warning::

        This is **experimental** and **likely to change**.
        If you need a more reliable or controllable hash,
        please `open an issue <https://github.com/p2p-ld/numpydantic/issues>`_.

    This hashes the bytes as stored, so it can be checked without zarr::

        cd array.zarr
        find . -type f | LC_ALL=C sort | xargs cat | sha256sum

    The same values stored with different chunking, compression, metadata,
    or zarr versions will have different hashes.
    """
    # zarr 3 only exposes the raw store contents through async methods
    prefix = f"{array.path}/" if array.path else ""
    prototype = default_buffer_prototype()

    async def _digest() -> str:
        keys = sorted([key async for key in array.store.list_prefix(prefix)])
        digest = hashlib.sha256()
        for key in keys:
            value = await array.store.get(key, prototype=prototype)
            digest.update(value.to_bytes())
        return digest.hexdigest()

    return sync(_digest())


def _parse_metadata(value: Any) -> ArrayMetadata:
    if isinstance(value, (ArrayV2Metadata, ArrayV3Metadata)):
        return value
    if not isinstance(value, dict) or value.get("zarr_format") not in (2, 3):
        raise ValueError("Not zarr array metadata")
    metadata_cls = ArrayV2Metadata if value["zarr_format"] == 2 else ArrayV3Metadata
    try:
        return metadata_cls.from_dict(value)
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid zarr array metadata: {e}") from e


@cache
def _metadata_dict_schema() -> core_schema.CoreSchema | None:
    """
    Schema for zarr's TypedDict of array metadata as JSON,
    or ``None`` if this version of pydantic can't read this version of zarr's types.
    """
    # zarr only imports JSON when type checking, so pydantic looks for it here instead
    JSON = Any  # noqa: F841
    try:
        with warnings.catch_warnings():
            # pydantic can't enforce zarr's ReadOnly fields, which is fine for a schema
            warnings.filterwarnings("ignore", message=".*ReadOnly.*")
            adapter = TypeAdapter(ArrayMetadataDict)
            # newer pydantic doesn't raise until the schema is used
            adapter.json_schema()
    except (PydanticUserError, NameError):
        return None
    return adapter.core_schema


class _MetadataJsonSchema:
    """Describe metadata with zarr's TypedDict if we can, or as any object if not"""

    def __get_pydantic_json_schema__(
        self, _schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = _metadata_dict_schema()
        if schema is None:
            return {"type": "object"}
        return handler(schema)


ZarrMetadata = Annotated[
    ArrayMetadata,
    GetPydanticSchema(
        lambda _type, _handler: core_schema.no_info_plain_validator_function(
            _parse_metadata,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda metadata: metadata.to_dict()
            ),
        )
    ),
    _MetadataJsonSchema(),
]
"""
Wrapped Metadata dataclasses from zarr, supporting roundtrip json serialization 
and json schema generation.

By themselves, pydantic can't build a schema from zarr's metadata dataclasses,
so use their own to_dict and from_dict methods instead.
Generating JSON Schema is similarly finnicky - 
we attempt to derive it from the TypedDict form of the schema, 
but a complex three-way interaction between python version, pydantic version, 
and zarr version makes that fail on older versions of all three.
So JSON schema for zarr metadata is "best effort" 
and falls back to being an unannotated `object` on failure. 
"""


class ZarrJsonDict(JsonDict):
    """Round-trip Json-able version of a Zarr Array"""

    info: ZarrMetadata
    """The array's metadata, in the format defined by the zarr spec"""
    type: Literal["zarr"]
    file: str | None = None
    path: str | None = None
    dtype: str | None = None
    hexdigest: str | None = None
    """
    Hash of the stored array, made by :func:`.hexdigest` when ``hash_array`` is
    ``True`` in the serialization context
    """
    value: list | None = None

    def to_array_input(self) -> ZarrArray | ZarrArrayPath:
        """
        Construct a ZarrArrayPath if file and path are present,
        otherwise a ZarrArray
        """
        if self.file:
            return ZarrArrayPath(file=self.file, path=self.path)
        dtype = str if self.dtype == "str" else self.dtype
        value = np.array(self.value, dtype=dtype)
        # zero-length dimensions are lost when dumped to nested lists
        if value.shape != self.info.shape:
            value = self.reshape_input(value, self.info.shape)
        return zarr.array(value, dtype=dtype)


class ZarrTyping(InterfaceTyping):
    """Static-typing companion for :class:`ZarrInterface`."""

    constructors = (
        ConstructorSpec(fullname="zarr.api.synchronous.zeros"),
        ConstructorSpec(fullname="zarr.api.synchronous.ones"),
        ConstructorSpec(fullname="zarr.api.synchronous.empty"),
        ConstructorSpec(fullname="zarr.api.synchronous.full"),
    )

    @classmethod
    def emit_imports(cls) -> list[str]:
        """import zarr and numpy!"""
        return ["import zarr"]

    @classmethod
    def emit_constructor_source(cls, shape: tuple[int, ...], dtype: str) -> str | None:
        """array constructor using :func:`zarr.zeros`"""
        return f"zarr.zeros({tuple(shape)!r}, dtype={dtype})"


class ZarrInterface(Interface):
    """
    Interface to in-memory or on-disk zarr arrays
    """

    name = "zarr"
    input_types = (ZarrArray, ZarrArrayPath)
    return_type = ZarrArray
    json_model = ZarrJsonDict
    typing = ZarrTyping

    @classmethod
    def enabled(cls) -> bool:
        """True if zarr is installed"""
        return ZarrArray is not None

    @staticmethod
    def _get_array(
        array: ZarrArray | str | dict | ZarrJsonDict | Path | ZarrArrayPath | Sequence,
    ) -> ZarrArray:
        if isinstance(array, ZarrArray):
            return array

        if isinstance(array, (str, Path)):
            array = ZarrArrayPath(file=array)
        elif isinstance(array, (tuple, list)):
            array = ZarrArrayPath.from_iterable(array)

        return array.open(mode="a")

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check if array is in-memory zarr array,
        a path to a zarr array, or a :class:`.ZarrArrayPath`
        """
        if isinstance(array, ZarrArray):
            return True

        if isinstance(array, dict):
            if array.get("type", False) == cls.name:
                return True
            # continue checking if dict contains a zarr file
            array = array.get("file", "")

        # See if can be coerced to ZarrArrayPath
        if isinstance(array, (Path, str)):
            array = ZarrArrayPath(file=array)

        if isinstance(array, (tuple, list)):
            # something that can be coerced to ZarrArrayPath
            with contextlib.suppress(ValueError):
                array = ZarrArrayPath.from_iterable(array)

        if isinstance(array, ZarrArrayPath):
            with contextlib.suppress(Exception):
                arr = array.open(mode="r")
                if isinstance(arr, ZarrArray):
                    return True

        return False

    def before_validation(
        self, array: ZarrArray | str | Path | ZarrArrayPath | Sequence
    ) -> ZarrArray:
        """
        Ensure that the zarr array is opened
        """
        return self._get_array(array)

    @classmethod
    def to_json(
        cls,
        array: ZarrArray | str | Path | ZarrArrayPath | Sequence,
        info: SerializationInfo | None = None,
    ) -> list | ZarrJsonDict:
        """
        Dump a Zarr Array to JSON

        If ``info.round_trip == False``, dump the array as a list of lists.
        This may be a memory-intensive operation.

        Otherwise, dump the array's metadata (see :attr:`zarr.Array.metadata`)
        as a :class:`.ZarrJsonDict`

        If the ``hash_array`` value in the context dictionary is ``True``,
        include a hash of the stored array from :func:`.hexdigest`

        If either the ``dump_array`` value in the context dictionary is ``True``
        or the zarr array is an in-memory array, dump the array as well
        (since without a persistent array it would be impossible to roundtrip and
        dumping to JSON would be meaningless)

        Passing ```dump_array': True`` to the serialization ``context``
        looks like this::

            model.model_dump_json(context={'zarr_dump_array': True})
        """
        array = cls._get_array(array)

        if info.round_trip:
            dump_array = False
            if info is not None and info.context is not None:
                dump_array = info.context.get("dump_array", False)
            is_file = False

            as_json = {"type": cls.name}
            as_json["dtype"] = "str" if array.dtype.type is str else array.dtype.name
            if isinstance(array.store, LocalStore):
                is_file = True
                as_json["file"] = str(array.store.root.absolute())
                as_json["path"] = array.path or None
            as_json["info"] = array.metadata
            if info.context is not None and info.context.get("hash_array", False):
                as_json["hexdigest"] = hexdigest(array)

            if dump_array or not is_file:
                as_json["value"] = array[:].tolist()

            as_json = ZarrJsonDict(**as_json)
        else:
            as_json = array[:].tolist()

        return as_json
