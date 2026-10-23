"""Mypy plugin for :class:`numpydantic.NDArray`.

The plugin enriches ``NDArray[Shape[...], dtype]`` annotations so that mypy
can statically reject:

- Invalid shape expressions (e.g. ``Shape["this is not valid"]``).
- Invalid dtype arguments
- Assignments between two ``NDArray`` annotations whose shapes or dtypes
  disagree.
- Functions whose declared ``NDArray`` return shape/dtype disagrees with the
  numpy constructor used in the function body

For NDArray-NDArray assignments this is straightforward.
For np.ndarray-NDArray and NDArray-np.ndarray assignments,
we attempt to enrich their type information from a set of constructors if possible.

Since this involves modifying the type information of types from other packages,
specifically those supported by an array interface like zarr or dask arrays,
non-numpy interfaces are disabled by default,
and must be explicitly enabled by adding them to the `[tool.numpydantic.mypy]` table:

```toml
[tool.numpydantic.mypy]
interfaces = [
  "zarr",
  "dask",
]
```

using the ``name`` attribute of the interface to enable.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from configparser import ConfigParser
from dataclasses import dataclass, field, fields
from tomllib import load as load_toml
from typing import Annotated, Any, Final, Literal, Self, get_args, get_origin

from annotated_types import Interval

from numpydantic.interface import ConstructorSpec, Interface

try:
    from mypy.nodes import (
        TypeAlias,
        TypeInfo,
    )
    from mypy.options import Options
    from mypy.plugin import (
        AnalyzeTypeContext,
        ClassDefContext,
        FunctionContext,
        MethodContext,
        Plugin,
        ReportConfigContext,
    )
    from mypy.types import (
        AnyType,
        CallableType,
        EllipsisType,
        Instance,
        LiteralType,
        ProperType,
        RawExpressionType,
        TupleType,
        Type,
        TypeOfAny,
        UnboundType,
        UnionType,
        UnpackType,
        get_proper_type,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("mypy must be installed to use the numpydantic plugin") from exc

from pydantic.mypy import PydanticPluginConfig

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.mypy.pydantic_ import _make_pydantic_hook
from numpydantic.mypy.ranges import RangeValue, module_has_ranges
from numpydantic.ndarray import NDArray
from numpydantic.shape import Range, Shape
from numpydantic.shape._variadic import _split_variadic
from numpydantic.vendor.nptyping.error import InvalidArgumentsError

CONFIGFILE_KEY = "numpydantic-mypy"
"""the table in an INI file where npd-specific settings are"""
NDARRAY_FULLNAME: Final = "numpydantic.ndarray.NDArray"
RANGE_FULLNAME: Final = "numpydantic.shape._range.Range"
LITERAL_FULLNAMES: Final = ("typing.Literal", "typing_extensions.Literal")
UNPACK_FULLNAMES: Final = ("typing.Unpack", "typing_extensions.Unpack")

BUILTIN_TO_NUMPY = {
    "builtins.float": "numpy.double",
    "builtins.int": "numpy.int_",
    "builtins.complex": "numpy.cdouble",
    "builtins.bool": "numpy.bool_",
    "builtins.str": "numpy.str_",
    "builtins.bytes": "numpy.bytes_",
    "datetime.datetime": "numpy.datetime64",
    "datetime.timedelta": "numpy.timedelta64",
    "builtins.object": "numpy.object_",
}


# ---------------------------------------------------------------------------
# Plugin entry-point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MypyPluginOptions:
    """
    Configure the mypy plugin.

    Set options in the [tool.numpydantic.mypy] table
    """

    interfaces: list[str] = field(default_factory=list)
    """
    A list of interface names that should have their constructor return types
    enriched (and replaced with np.ndarray types).
    
    Numpy constructors are always enriched (to disable them, disable the plugin). 
    """

    @classmethod
    def from_options(cls, options: Options) -> Self:
        """Load from mypy's options object, which refers to the active toml file"""
        # borrowing from https://github.com/pydantic/pydantic/blob/a20c0ee267150c3bb0f82bf05e0806fa65b1e70c/pydantic/mypy.py#L231

        kwargs: dict[str, Any] = {}
        if options.config_file.endswith(".toml"):
            with open(options.config_file, "rb") as f:
                toml_config = load_toml(f)

            kwargs = toml_config.get("tool", {}).get("numpydantic", {}).get("mypy", {})
        elif options.config_file:
            # an .ini file
            parser = ConfigParser()
            parser.read(options.config_file)
            if CONFIGFILE_KEY in parser:
                kwargs = dict(parser[CONFIGFILE_KEY])
                for f in fields(cls):
                    if f.name not in kwargs:
                        continue
                    # janky, but our config isn't complex enough to
                    # write a whole ass parser.
                    # or a whole arse parser, if you're english.
                    # arseparser. is this anything.
                    if f.type.startswith("list"):
                        kwargs[f.name] = [
                            item.strip()
                            for item in kwargs[f.name].split(",")
                            if item.strip()
                        ]

        return cls(**kwargs)


class NumpydanticMypyPlugin(Plugin):
    """Static type checking for ``numpydantic.NDArray``."""

    def __init__(self, options: Options):
        super().__init__(options)
        self.config = MypyPluginOptions.from_options(options)
        self.pydantic_config = PydanticPluginConfig(options)
        self._functions: dict[str, ConstructorSpec] = {}
        self._methods: dict[str, ConstructorSpec] = {}
        self._interface_inputs: list[str] = []
        self._load_interfaces()

    def get_type_analyze_hook(
        self, fullname: str
    ) -> Callable[[AnalyzeTypeContext], Type] | None:
        """Convert an NDArray annotation to an enriched np.ndarray annotation"""
        if fullname == NDARRAY_FULLNAME:
            return _analyze_ndarray_type
        return None

    def get_function_hook(
        self, fullname: str
    ) -> Callable[[FunctionContext], Type] | None:
        """Enrich the np.ndarray annotation from a supported array constructor"""
        from numpydantic.mypy.inference import (
            _make_function_hook,
        )

        spec = self._functions.get(fullname)
        if spec is not None:
            return _make_function_hook(spec)
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        """
        Enrich the np.ndarray annotation from a supported array constructor,
        except for if it's a method.
        """
        from numpydantic.mypy.inference import _make_method_hook

        spec = self._methods.get(fullname)
        if spec is not None:
            return _make_method_hook(spec)
        return None

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        """
        Extend how pydantic annotates its classes
        to allow for converting additional input types to NDArrays
        """
        sym = self.lookup_fully_qualified(fullname)
        if (
            sym
            and isinstance(sym.node, TypeInfo)
            and sym.node.has_base("pydantic.main.BaseModel")
        ):
            return _make_pydantic_hook(self.pydantic_config, self._interface_inputs)

    def report_config_data(self, ctx: ReportConfigContext) -> Any:
        """
        Make mypy recheck modules with shape ranges on every run,
        rather than loading them from its cache.

        mypy stores literal values in its cache as plain data,
        so a cached :class:`.RangeValue` comes back as a plain ``str``
        that no longer does range checking.

        mypy rechecks a module when the value returned here differs from the one
        stored in its cache, so we store ``True`` for modules with ranges when
        writing the cache, and always return ``None`` when checking it.
        """
        if ctx.is_check or self._modules is None:
            return None
        module = self._modules.get(ctx.id)
        if module is not None and module_has_ranges(module):
            return True
        return None

    def _load_interfaces(self) -> None:
        enabled_interfaces = set(self.config.interfaces + ["numpy"])

        for iface in Interface.interfaces(with_disabled=True):
            if iface.name not in enabled_interfaces:
                continue
            typing_cls = iface.typing
            if typing_cls is None:
                continue
            for spec in typing_cls.constructors:
                if spec.mode == "method":
                    self._methods[spec.fullname] = spec
                else:
                    self._functions[spec.fullname] = spec

        self._interface_inputs = _load_interface_input_fullnames()


def plugin(version: str) -> type[NumpydanticMypyPlugin]:  # noqa: ARG001
    """Mypy plugin entry-point."""
    return NumpydanticMypyPlugin


# ---------------------------------------------------------------------------
# NDArray type-analyze hook
# ---------------------------------------------------------------------------


def _analyze_ndarray_type(ctx: AnalyzeTypeContext) -> Type:
    """
    Render ``NDArray[shape, dtype]`` as a static numpy.ndarray type.
    """
    args = ctx.type.args

    if len(args) == 0:
        shape_arg, dtype_arg = None, AnyType(TypeOfAny.special_form)
    elif len(args) == 1:
        shape_arg, dtype_arg = args[0], AnyType(TypeOfAny.special_form)
    elif len(args) == 2:
        shape_arg, dtype_arg = args
    else:
        ctx.api.fail(
            f"NDArray expects at most 2 type arguments, got {len(args)}", ctx.context
        )
        return AnyType(TypeOfAny.from_error)

    shape = _parse_shape_arg(shape_arg, ctx)
    dtype = _parse_dtype_arg(dtype_arg, ctx)
    ndarray_type = _build_ndarray_type(ctx, shape=shape, dtype=dtype)
    return ndarray_type


def _build_ndarray_type(
    ctx: AnalyzeTypeContext | FunctionContext | MethodContext,
    shape: ProperType | None,
    dtype: ProperType,
) -> Type:
    """
    Build the rendered ``NDArray`` type as its final np.ndarray form
    """
    api = ctx.api

    if shape is None:
        shape = AnyType(TypeOfAny.special_form)

    if isinstance(ctx, AnalyzeTypeContext):
        dtype_instance = api.named_type("numpy.dtype", [dtype])
        numpy_variant = api.named_type("numpy.ndarray", [shape, dtype_instance])
    else:
        dtype_instance = api.named_generic_type("numpy.dtype", [dtype])
        numpy_variant = api.named_generic_type("numpy.ndarray", [shape, dtype_instance])
    numpy_variant.type.metadata = {"numpydantic": True}
    return numpy_variant


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------


def _parse_shape_arg(arg: Type | None, ctx: AnalyzeTypeContext) -> TupleType | None:
    """
    Parse the shape constraint into its numpy representation,
    or ``None`` if the shape is unconstrained or invalid.

    The annotation is converted into the runtime object it stands for,
    normalized by ``NDArray`` the same way it is at runtime,
    and then the canonical tuple form is converted into mypy types.
    """
    if arg is None:
        return None
    try:
        with warnings.catch_warnings():
            # deprecated forms warn at runtime, not while type checking
            warnings.simplefilter("ignore")
            shape = _to_runtime(arg, ctx)
            canonical = NDArray[shape, Any].__args__[0]
        if canonical is Any:
            return None
        return _canonical_to_type(canonical, ctx)
    except (InvalidShapeExpressionError, InvalidArgumentsError) as exc:
        ctx.api.fail(str(exc), ctx.context)
        return None


def _to_runtime(typ: Type, ctx: AnalyzeTypeContext) -> Any:
    """The runtime object that an unanalyzed shape annotation stands for"""
    typ = get_proper_type(typ)
    if isinstance(typ, RawExpressionType):
        return typ.literal_value
    elif isinstance(typ, EllipsisType):
        return ...
    elif isinstance(typ, UnpackType):
        return next(iter(_to_runtime(typ.type, ctx)))
    elif isinstance(typ, UnboundType):
        fullname = _resolve_fullname(typ, ctx)
        if fullname in LITERAL_FULLNAMES:
            return Literal[tuple(_to_runtime(arg, ctx) for arg in typ.args)]
        elif fullname in UNPACK_FULLNAMES and len(typ.args) == 1:
            return next(iter(_to_runtime(typ.args[0], ctx)))
        elif fullname == RANGE_FULLNAME:
            return Range[tuple(_to_runtime(arg, ctx) for arg in typ.args)]
        elif fullname == "builtins.tuple" and typ.args:
            # Shape is tuple to type checkers, and Shape[...] normalizes tuple args
            return Shape[tuple(_to_runtime(arg, ctx) for arg in typ.args)]

    return _analyzed_to_runtime(get_proper_type(ctx.api.analyze_type(typ)), typ)


def _resolve_fullname(typ: UnboundType, ctx: AnalyzeTypeContext) -> str | None:
    """
    The fully qualified name of what an unanalyzed type refers to,
    following aliases like ``L = Literal``
    """
    sym = ctx.api.lookup_qualified(typ.name, typ, suppress_errors=True)
    if sym is None or sym.node is None:
        return None
    node = sym.node
    if isinstance(node, TypeAlias) and node.no_args:
        target = get_proper_type(node.target)
        if isinstance(target, Instance):
            return target.type.fullname
    return node.fullname


def _analyzed_to_runtime(typ: ProperType, original: Type) -> Any:
    """The runtime object for a type that mypy has analyzed, like ``int``"""
    if isinstance(typ, AnyType):
        return Any
    elif isinstance(typ, LiteralType):
        return Literal[typ.value]
    elif isinstance(typ, Instance):
        if typ.type.fullname == "builtins.int":
            return int
        elif typ.type.fullname == "builtins.tuple" and len(typ.args) == 1:
            # a variadic tuple, like AnyDims
            item = _analyzed_to_runtime(get_proper_type(typ.args[0]), original)
            return tuple[item, ...]
    elif isinstance(typ, TupleType):
        items = (_analyzed_to_runtime(get_proper_type(t), original) for t in typ.items)
        return tuple[tuple(items)]
    raise InvalidShapeExpressionError(f"Invalid shape dimension: {original}")


def _canonical_to_type(canonical: Any, ctx: AnalyzeTypeContext) -> TupleType:
    """Convert a canonical runtime shape like ``tuple[L[3], *AnyDims]`` to mypy"""
    int_instance = ctx.api.named_type("builtins.int", [])
    fixed, tail = _split_variadic(get_args(canonical))
    items = [_canonical_dim_to_type(dim, int_instance) for dim in fixed]
    if tail is not None:
        repeated = _canonical_dim_to_type(tail, int_instance)
        items.append(UnpackType(ctx.api.named_type("builtins.tuple", [repeated])))
    return TupleType(
        items, fallback=ctx.api.named_type("builtins.tuple", [int_instance])
    )


def _canonical_dim_to_type(dim: Any, int_instance: Instance) -> Type:
    """Convert one dimension of a canonical runtime shape to mypy"""
    if dim is int:
        return int_instance
    elif dim is Any:
        return AnyType(TypeOfAny.special_form)

    origin = get_origin(dim)
    if origin is Literal and len(get_args(dim)) == 1:
        return LiteralType(get_args(dim)[0], fallback=int_instance)
    elif origin is Annotated:
        base, *metadata = get_args(dim)
        for meta in metadata:
            if isinstance(meta, Interval):
                return LiteralType(RangeValue(meta), fallback=int_instance)
        # other metadata, like labels, doesn't change the type
        return _canonical_dim_to_type(base, int_instance)
    raise InvalidShapeExpressionError(f"Invalid shape dimension: {dim!r}")


# ---------------------------------------------------------------------------
# Dtype parsing
# ---------------------------------------------------------------------------


def _parse_dtype_arg(arg: Type, ctx: AnalyzeTypeContext) -> ProperType:
    """
    Return a ProperType representing the dtype constraint.

    Accepts variants of numpy dtypes and builtin types,
    and if none can be found, treat it like numpy.object_
    """
    analyzed = ctx.api.analyze_type(arg)
    analyzed = get_proper_type(analyzed)
    return _dtype_as_numpy(analyzed, ctx)


def _dtype_as_numpy(
    t: ProperType, ctx: AnalyzeTypeContext | FunctionContext | MethodContext
) -> ProperType:
    """Cast the dtype arg into a numpy generic type that can be used in ndarray"""
    # first part, unwrapping containers, either recursively or falling through
    if isinstance(t, AnyType):
        return t
    elif isinstance(t, UnionType):
        items = [_dtype_as_numpy(get_proper_type(item), ctx) for item in t.items]
        return UnionType.make_union(items)
    elif isinstance(t, TupleType):
        # Tuple-as-union shorthand: NDArray[shape, (a, b)] - rare in static
        # type annotations but valid at runtime. Treat as union.
        items = [_dtype_as_numpy(get_proper_type(item), ctx) for item in t.items]
        return UnionType.make_union(items)
    elif isinstance(t, CallableType):
        t = get_proper_type(t.ret_type)

    # second part,
    if isinstance(t, Instance):
        if not _is_numpy_generic(t.type):
            numpy_equivalent = BUILTIN_TO_NUMPY.get(
                t.type.fullname, BUILTIN_TO_NUMPY["builtins.object"]
            )
            if isinstance(ctx, AnalyzeTypeContext):
                symbol = ctx.api.lookup_fully_qualified(numpy_equivalent)
            else:
                symbol = ctx.api.lookup_qualified(numpy_equivalent)
            node = symbol.node
            if isinstance(node, TypeAlias):
                t = node.target
            else:
                t = symbol.node.type_object_type.ret_type

        return t
    elif isinstance(t, LiteralType):
        ctx.api.fail(
            f"invalid dtype Literal[{t.value!r}]; expected a numpy generic",
            ctx.context,
        )
        return AnyType(TypeOfAny.from_error)
    ctx.api.fail("invalid dtype argument", ctx.context)
    return AnyType(TypeOfAny.from_error)


# --------------------------------------------------
# Etcetera
# --------------------------------------------------


def _is_numpy_generic(info: TypeInfo) -> bool:
    """Whether a TypeInfo derives from numpy.generic."""
    return any(base.fullname == "numpy.generic" for base in info.mro)


def _load_interface_input_fullnames() -> list[str]:
    """
    Get all allowable input types defined by interfaces,
    used when NDArray is declared within a pydantic model.

    This does *not* filter using the `pyproject.toml` config's
    enabled interfaces list - that is for controlling whether
    we modify the types from those interfaces when enriching constructors,
    this is for determining what input types are allowed
    to a pydantic model.

    filter out builtin inputs types, if any exist -
    they are too generic, and mypy plugin should enforce stricter typing habits.
    All the builtins that are allowed at runtime like a string for VideoInterface
    have typed counterparts, like H5ArrayPath and ZarrArrayPath.
    """
    fullnames = []
    for interface in Interface.interfaces(with_disabled=False):
        inputs = interface.input_types
        fullnames.extend(
            f"{t.__module__}.{t.__qualname__}"
            for t in inputs
            if t.__module__ != "builtins"
        )

    fullnames = set(fullnames)

    # FIXME: bit of a hack - pathlib's actual loc is pathlib._local.Path
    # but mypy can't find this name
    if "pathlib._local.Path" in fullnames:
        fullnames.discard("pathlib._local.Path")
        fullnames.add("pathlib.Path")

    # we don't allow the generic Any form of numpy ndarray,
    # a spec'd numpy.ndarray type is union'd to these types
    fullnames = list(set(fullnames) - {"numpy.ndarray"})
    return fullnames
