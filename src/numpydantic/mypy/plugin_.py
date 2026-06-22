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
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

if sys.version_info < (3, 11):
    from tomli import load as load_toml
    from typing_extensions import Self
else:
    from typing import Self

    from tomllib import load as load_toml

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
    )
    from mypy.types import (
        AnyType,
        CallableType,
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

from numpydantic.mypy.pydantic_ import _make_pydantic_hook
from numpydantic.validation.shape import (
    _is_range,
    validate_shape_expression,
)
from numpydantic.vendor.nptyping.error import InvalidShapeError
from numpydantic.vendor.nptyping.shape_expression import (
    get_dimensions,
    remove_labels,
)

NDARRAY_FULLNAME: Final = "numpydantic.ndarray.NDArray"
_NUMERIC_RE = re.compile(r"^[0-9]+$")

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
        if options.config_file is None:
            return MypyPluginOptions()

        with open(options.config_file, "rb") as f:
            toml_config = load_toml(f)

        if toml_config is None:
            return MypyPluginOptions()

        toml_options = (
            toml_config.get("tool", {}).get("numpydantic", {}).get("mypy", {})
        )
        return MypyPluginOptions(**toml_options)


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


class RangeValue(str):
    """
    An unholy python type used to overcome the inability to declare custom mypy types
    for checking ranges, otherwise we would have to do something like
    Union[Literal[1], Literal[2], Literal[3], ...]

    See: https://github.com/python/mypy/issues/16497#issuecomment-4570099557
    """

    def __new__(cls, low: int | str, high: int | str):  # noqa: D102
        inst = super(cls, cls).__new__(cls, f"{low}-{high}")
        inst.low = low
        inst.high = high
        inst.low_wildcard = isinstance(low, str) and low == "*"
        inst.high_wildcard = isinstance(high, str) and high == "*"
        return inst

    def __hash__(self):
        return hash((self.low, self.high))

    def __eq__(self, other: Any) -> bool:
        low_valid = False
        high_valid = False
        if isinstance(other, str) and _is_range(other):
            # TODO: figure out how to check if one range falls within another
            # can't tell if we're on the left or right side since associativity
            return True
        elif isinstance(other, int):
            if self.low_wildcard or self.low <= other:
                low_valid = True
            if self.high_wildcard or self.high >= other:
                high_valid = True
        else:
            raise TypeError("Can only compare integers or ranges to ranges")

        return low_valid and high_valid


def plugin(version: str) -> type[NumpydanticMypyPlugin]:  # noqa: ARG001
    """Mypy plugin entry-point."""
    return NumpydanticMypyPlugin


# ---------------------------------------------------------------------------
# NDArray type-analyze hook
# ---------------------------------------------------------------------------


def _analyze_ndarray_type(ctx: AnalyzeTypeContext) -> Type:
    """
    Render ``NDArray[shape, dtype]`` as a static numpy.ndarray type.

    TODO: This is a *bit* circular,
    we build up args into a string since there are several forms,
    then break it back down again into a tuple declaration.
    ideally we just go everything -> tuple rather than roundtripping through string,
    but hey just getting the thing working for now, it's been a hack.
    There are some things that the python type system still can't do,
    but e.g. we aren't using TypeVarTuple even.
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


def _parse_shape_arg(arg: Type, ctx: AnalyzeTypeContext) -> TupleType | None:
    """
    Parse the shape constraint into its numpy representation
    """
    expr = _shape_expression_from_arg(arg, ctx)
    if expr is None:
        return None
    try:
        validate_shape_expression(expr)
    except InvalidShapeError as exc:
        ctx.api.fail(str(exc), ctx.context)
        return None

    dims = remove_labels(get_dimensions(expr))
    dims = [d.strip() for d in dims if d.strip()]
    return _dims_to_tuple(dims, ctx)


def _shape_expression_from_arg(arg: Type, ctx: AnalyzeTypeContext) -> str | None:
    """Walk an ``UnboundType`` to recover the original shape-expression string."""
    arg = get_proper_type(arg)

    if isinstance(arg, AnyType):
        return None

    if isinstance(arg, RawExpressionType):
        # strings, or strings inside Literal[]
        return str(arg.literal_value)

    if isinstance(arg, UnboundType):
        if not arg.args:
            return None

        if len(arg.args) == 1:
            return _shape_expression_from_arg(arg.args[0], ctx)
        # Shape[3, 3, ...] varargs
        return ", ".join(str(arg) for arg in arg.args)

    return None


def _dims_to_tuple(
    dims: list[str | int], ctx: AnalyzeTypeContext
) -> TupleType | Instance:
    """Combine dimensions into their tuple form"""
    api = ctx.api
    int_instance = api.named_type("builtins.int", [])
    variadic_tuple = api.named_type("builtins.tuple", [int_instance])
    shape_type: TupleType
    if dims is None:
        return variadic_tuple
    else:
        items: list[Type] = []
        trailing_variadic = False
        for dim in dims:
            if dim == "...":
                trailing_variadic = True
                continue
            items.append(_dim_to_type(dim, int_instance))

        if trailing_variadic:
            unpack = UnpackType(api.named_type("builtins.tuple", [int_instance]))
            shape_type = TupleType(items + [unpack], fallback=variadic_tuple)
        else:
            shape_type = TupleType(items, fallback=variadic_tuple)
    return shape_type


def _dim_to_type(dim: str, int_instance: Instance) -> Type:
    """Convert a single dim string to its mypy Type representation."""
    if _NUMERIC_RE.match(dim):
        return LiteralType(int(dim), fallback=int_instance)
    elif _is_range(dim):
        parts = dim.split("-")
        low = int(parts[0]) if parts[0] != "*" else parts[0]
        high = int(parts[1]) if parts[1] != "*" else parts[1]
        typ = LiteralType(RangeValue(low, high), int_instance)
        return typ
    elif dim == "*":
        return int_instance
    else:
        raise NotImplementedError("Unhandled dim, howd you get here!")


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
