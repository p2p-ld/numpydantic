"""
Shape typevars in the mypy plugin.

Type checkers solve a type variable as the narrowest type that fits every argument,
so passing sizes 3 and 4 for the same shape typevar gives ``int``,
and the call is accepted.
A shape typevar means one size, so calls where the arguments disagree are refused.
Variables constrained to single sizes, like ``TypeVar("N", L[3], L[4])``,
are left to mypy, which already refuses them.
"""

from collections.abc import Iterator

from mypy import errorcodes as codes
from mypy.constraints import SUPERTYPE_OF, infer_constraints
from mypy.maptype import map_instance_to_supertype
from mypy.nodes import (
    Decorator,
    DictExpr,
    Expression,
    FuncBase,
    ListExpr,
    StarExpr,
    SymbolNode,
    TypeInfo,
    Var,
)
from mypy.plugin import CheckerPluginInterface, FunctionSigContext, MethodSigContext
from mypy.type_visitor import TypeQuery
from mypy.typeops import get_type_vars
from mypy.types import (
    AnyType,
    CallableType,
    FunctionLike,
    Instance,
    LiteralType,
    TupleType,
    Type,
    TypeOfAny,
    TypeVarId,
    TypeVarLikeType,
    TypeVarType,
    UnpackType,
    get_proper_type,
)


class _ShapeTypeVars(TypeQuery[set[TypeVarId]]):
    """
    The type variables used as sizes in the ndarray shapes within a type::

        NDArray[Shape[N, N], np.float64]
            -> {N}
        list[NDArray[Shape[N, *tuple[M, ...]], np.float64]]
            -> {N, M}
        dict[T, NDArray[Shape[L[3]], np.float64]]
            -> set(), since T isn't a size
    """

    def strategy(self, items: list[set[TypeVarId]]) -> set[TypeVarId]:
        return set().union(*items)

    def visit_instance(self, t: Instance, /) -> set[TypeVarId]:
        found = self.query_types(t.args)
        if t.type.fullname == "numpy.ndarray" and t.args:
            shape = get_proper_type(t.args[0])
            if isinstance(shape, TupleType):
                for item in map(get_proper_type, shape.items):
                    if isinstance(item, UnpackType):
                        # a repeated size, like *tuple[N, ...]
                        unpacked = get_proper_type(item.type)
                        if isinstance(unpacked, Instance) and unpacked.args:
                            item = get_proper_type(unpacked.args[0])
                    if isinstance(item, TypeVarType):
                        found.add(item.id)
        return found


def has_shape_typevars(node: SymbolNode | None) -> bool:
    """
    Whether any shape in the signature of a function, method,
    or class (its ``__init__`` or ``__new__``) has a typevar in it
    """
    nodes: list[SymbolNode | None]
    if isinstance(node, TypeInfo):
        nodes = [node.get_method("__init__"), node.get_method("__new__")]
    else:
        nodes = [node]

    types: list[Type | None] = []
    for item in nodes:
        if isinstance(item, Decorator):
            types.extend([item.var.type, item.func.type])
        elif isinstance(item, (FuncBase, Var)):
            types.append(item.type)
    return any(t is not None and t.accept(_ShapeTypeVars()) for t in types)


def check_shape_typevars(ctx: FunctionSigContext | MethodSigContext) -> FunctionLike:
    """
    Check that all sizes given to typevars are equal.

    For example, with this function::

        N = TypeVar("N", bound=int)

        def scale_rows(
            matrix: NDArray[Shape[N, N], np.float64],
            rows: list[NDArray[Shape[N], np.float64]],
        ) -> None: ...

    this call is refused, since ``N`` is 3 in ``matrix``
    but 4 in the second of the ``rows``::

        scale_rows(np.zeros((3, 3)), [np.zeros((3,)), np.zeros((4,))])
        # error: Shape typevar "N" is 3 in "matrix" but 4 in "rows"

    The examples in the functions below use this function and call.
    """
    shape_vars = _checked_typevars(ctx.default_signature)
    sizes: dict[TypeVarId, tuple[int, str]] = {}
    for arg_name, annotation, actual in _arguments(ctx, shape_vars):
        for var, size in _sizes(annotation, actual, shape_vars):
            first_size, first_name = sizes.setdefault(var.id, (size, arg_name))
            if size != first_size:
                ctx.api.fail(
                    _conflict_message(var.name, first_size, first_name, size, arg_name),
                    ctx.context,
                    code=codes.ARG_TYPE,
                )
                # stop checking it, so it's only reported once
                shape_vars.discard(var.id)
    return ctx.default_signature


def _checked_typevars(signature: CallableType) -> set[TypeVarId]:
    """
    The type variables used as sizes in the signature's shapes, except those
    constrained to single sizes, which mypy already keeps to one size.

    The signature of ``scale_rows`` (see ``check_shape_typevars``) gives ``{N}``.
    This one also gives ``{N}``, since ``C`` is left to mypy and ``T`` isn't a size::

        C = TypeVar("C", L[3], L[4])
        T = TypeVar("T")

        def other(a: NDArray[Shape[N, C], np.float64], b: T) -> None: ...
    """
    used = set().union(
        *(formal.accept(_ShapeTypeVars()) for formal in signature.arg_types)
    )
    return used - {var.id for var in signature.variables if _single_size(var)}


def _arguments(
    ctx: FunctionSigContext | MethodSigContext, shape_vars: set[TypeVarId]
) -> Iterator[tuple[str, Type, Type]]:
    """
    The type of each argument, with the name and type of its parameter,
    for parameters that use any of ``shape_vars``.
    Each item of a list or dict display counts as its own argument, see _per_item.

    For the call in ``check_shape_typevars``::

        scale_rows(np.zeros((3, 3)), [np.zeros((3,)), np.zeros((4,))])

    this gives one tuple for ``matrix``, and one for each item of ``rows``::

        (
            "matrix",
            NDArray[Shape[N, N], np.float64],
            NDArray[Shape[L[3], L[3]], Any],
        )
        (
            "rows",
            list[NDArray[Shape[N], np.float64]],
            list[NDArray[Shape[L[3]], Any]],
        )
        (
            "rows",
            list[NDArray[Shape[N], np.float64]],
            list[NDArray[Shape[L[4]], Any]],
        )
    """
    signature = ctx.default_signature
    for i, (annotation, actuals) in enumerate(zip(signature.arg_types, ctx.args)):
        if not any(var.id in shape_vars for var in get_type_vars(annotation)):
            continue
        name = signature.arg_names[i] or str(i + 1)
        # one parameter can take several arguments, like *args
        for actual in actuals:
            for typ in _expanded_items(actual, ctx.api):
                yield name, annotation, typ


def _sizes(
    annotation: Type, actual: Type, shape_vars: set[TypeVarId]
) -> Iterator[tuple[TypeVarLikeType, int]]:
    """
    The sizes an argument gives to ``shape_vars``,
    paired up by mypy the same way as when it solves them.

    mypy walks the parameter's type and the argument's type together,
    and gives a size wherever a type variable meets a literal size.
    For the tuples from ``_arguments``::

        (NDArray[Shape[N, N], np.float64], NDArray[Shape[L[3], L[3]], Any])
            -> (N, 3), (N, 3)
        (list[NDArray[Shape[N], np.float64]], list[NDArray[Shape[L[4]], Any]])
            -> (N, 4)

    A size that isn't known, like ``int``, gives nothing::

        (NDArray[Shape[N, M], np.float64], NDArray[Shape[L[3], int], Any])
            -> (N, 3)
    """
    for constraint in infer_constraints(annotation, actual, SUPERTYPE_OF):
        size = get_proper_type(constraint.target)
        # ranges are literals too, but with strings for values
        if (
            constraint.type_var in shape_vars
            and isinstance(size, LiteralType)
            and type(size.value) is int
        ):
            yield constraint.origin_type_var, size.value


def _expanded_items(expr: Expression, api: CheckerPluginInterface) -> Iterator[Type]:
    """
    The argument's type once for each item of a list or dict display,
    since mypy joins the types of the items into one, like 3 and 4 into int.

    Any other argument gives just its type::

        np.zeros((3,))
            -> NDArray[Shape[L[3]], Any]

    A list gives one type per item, where mypy's own type for this list
    is ``list[NDArray[Shape[int], Any]]``::

        [np.zeros((3,)), np.zeros((4,))]
            -> list[NDArray[Shape[L[3]], Any]]
            -> list[NDArray[Shape[L[4]], Any]]

    and so do dicts, nested lists, and unpacked items::

        {"a": np.zeros((3,)), "b": np.zeros((4,))}
            -> dict[str, NDArray[Shape[L[3]], Any]]
            -> dict[str, NDArray[Shape[L[4]], Any]]

        [[np.zeros((3,))], [np.zeros((4,))]]
            -> list[list[NDArray[Shape[L[3]], Any]]]
            -> list[list[NDArray[Shape[L[4]], Any]]]

        fours: list[NDArray[Shape[L[4]], Any]]
        [np.zeros((3,)), *fours]
            -> list[NDArray[Shape[L[3]], Any]]
            -> list[NDArray[Shape[L[4]], Any]]
    """
    typ = get_proper_type(api.get_expression_type(expr))
    if isinstance(expr, ListExpr):
        unpacked_as = "typing.Iterable"
        items = [
            (item.expr, True) if isinstance(item, StarExpr) else (item, False)
            for item in expr.items
        ]
    elif isinstance(expr, DictExpr):
        unpacked_as = "typing.Mapping"
        # a None key is **unpacking
        items = [(value, key is None) for key, value in expr.items]
    else:
        items = []
    if not items or not isinstance(typ, Instance) or not typ.args:
        yield typ
        return

    # the item type is the last type argument, as in list[X] and dict[K, X]
    for item, unpacked in items:
        if unpacked:
            item_types = [_unpacked_item_type(item, unpacked_as, api)]
        else:
            item_types = list(_expanded_items(item, api))
        for item_type in item_types:
            yield typ.copy_modified(args=[*typ.args[:-1], item_type])


def _unpacked_item_type(
    expr: Expression, unpacked_as: str, api: CheckerPluginInterface
) -> Type:
    """
    The type of the items of an unpacked iterable, like *xs,
    or of the values of an unpacked mapping, like **d::

        threes: list[NDArray[Shape[L[3]], np.float64]]
        fours: dict[str, NDArray[Shape[L[4]], np.float64]]

        _unpacked_item_type(threes, "typing.Iterable")
            -> NDArray[Shape[L[3]], np.float64]
        _unpacked_item_type(fours, "typing.Mapping")
            -> NDArray[Shape[L[4]], np.float64]
    """
    typ = get_proper_type(api.get_expression_type(expr))
    container = api.named_generic_type(unpacked_as, []).type
    if isinstance(typ, Instance) and typ.type.has_base(container.fullname):
        return map_instance_to_supertype(typ, container).args[-1]
    return AnyType(TypeOfAny.special_form)


def _single_size(var: Type) -> bool:
    """
    Whether a type variable is constrained to single sizes, like L[3] and L[4],
    which mypy already keeps to one size. Constraints like int, a Range,
    or L[1, 2] can each fit several sizes, so they're checked here::

        TypeVar("N", L[3], L[4])     -> True
        TypeVar("N", int, L[3])      -> False
        TypeVar("N", L[1, 2], L[5])  -> False
        TypeVar("N", bound=int)      -> False
    """
    return (
        isinstance(var, TypeVarType)
        and bool(var.values)
        and all(isinstance(get_proper_type(value), LiteralType) for value in var.values)
    )


def _conflict_message(
    var: str, first_size: int, first_name: str, size: int, name: str
) -> str:
    if first_name == name:
        return f'Shape typevar "{var}" is both {first_size} and {size} in "{name}"'
    return (
        f'Shape typevar "{var}" is {first_size} in "{first_name}" '
        f'but {size} in "{name}"'
    )
