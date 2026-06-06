"""
Use interface type declaration objects to infer shape/dtype data
"""

from __future__ import annotations

from collections.abc import Callable

from mypy.nodes import Expression, IntExpr, ListExpr, TupleExpr
from mypy.plugin import DynamicClassDefContext, FunctionContext, MethodContext
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    TupleType,
    Type,
    TypeOfAny,
    get_proper_type,
)

from numpydantic.interface import ConstructorSpec
from numpydantic.mypy.plugin_ import (
    _build_ndarray_type,
    _dtype_as_numpy,
)


def _make_function_hook(
    spec: ConstructorSpec,
) -> Callable[[FunctionContext], Type]:
    def hook(ctx: FunctionContext) -> Type:
        return _refine_constructor_return(ctx, spec)

    return hook


def _make_method_hook(
    spec: ConstructorSpec,
) -> Callable[[MethodContext], Type]:
    def hook(ctx: MethodContext) -> Type:
        return _refine_constructor_return(ctx, spec)

    return hook


def _refine_constructor_return(
    ctx: FunctionContext | MethodContext | DynamicClassDefContext,
    spec: ConstructorSpec,
) -> Type:
    """
    Refine a return value into an np.ndarray type with
    specific type when the shape and/or dtype call args are literals.

    ``spec`` describes where to look for the shape and dtype call expressions
    (positional index vs. keyword name).
    """
    if not ctx.args:
        return ctx.api.named_generic_type("numpy.ndarray", [])

    int_instance = ctx.api.named_generic_type("builtins.int", [])

    shape_expr = _find_arg(ctx.args, ctx.arg_names, spec.shape_arg)
    if shape_expr is not None and isinstance(shape_expr, TupleExpr):
        # TODO: forwarding shape inference from e.g. indexing array.shape
        literal_dims = _literal_dims_from_expr(shape_expr, int_instance)
        shape = TupleType(
            literal_dims,
            fallback=int_instance,
        )
    else:
        shape = AnyType(TypeOfAny.special_form)

    dtype_expr = _find_arg(ctx.args, ctx.arg_names, spec.dtype_arg)
    if dtype_expr is not None:
        dtype = _dtype_as_numpy(
            get_proper_type(ctx.api.get_expression_type(dtype_expr)), ctx
        )
    else:
        dtype = AnyType(TypeOfAny.special_form)

    return _build_ndarray_type(ctx, shape, dtype)


def _find_arg(
    args: list[list[Expression]],
    arg_names: list[list[str | None]],
    arg: int | str,
) -> Expression | None:
    """Resolve a shape argument by positional index or keyword name."""
    # flatten args
    args = [inner for outer in args for inner in outer]

    if isinstance(arg, int):
        if 0 <= arg < len(args) and args[arg]:
            return args[arg]
        return None
    else:
        # flatten arg_names
        flat_args = [inner for outer in arg_names for inner in outer]

        for i, candidate in enumerate(flat_args):
            if candidate == arg and i < len(args) and args[i]:
                return args[i]


def _literal_dims_from_expr(
    expr: Expression, int_instance: Instance
) -> list[LiteralType | Instance] | None:
    """
    If ``expr`` is a literal tuple/list of integers, return them.

    Returns ``None`` for non-literal shapes, mixed types, or scalar-int shapes
    (the 1-D form ``np.empty(5)``).
    """
    if isinstance(expr, (TupleExpr, ListExpr)):
        dims: list[LiteralType | Instance] = [
            (
                LiteralType(item.value, fallback=int_instance)
                if isinstance(item, IntExpr)
                else int_instance
            )
            for item in expr.items
        ]
        return dims
    elif isinstance(expr, IntExpr):
        # 1-D constructor: np.empty(5) -> tuple[Literal[5]]
        return [expr.value]
    return None
