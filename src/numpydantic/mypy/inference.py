"""
Use interface type declaration objects to infer shape/dtype data
"""

from __future__ import annotations

from collections.abc import Callable

from mypy.nodes import Expression, IntExpr, ListExpr, TupleExpr, UnaryExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import Instance, LiteralType, TupleType, Type, get_proper_type

from numpydantic.interface import ConstructorSpec
from numpydantic.mypy.plugin_ import (
    BUILTIN_TO_NUMPY,
    _is_numpy_generic,
)


def _make_function_hook(
    spec: ConstructorSpec,
) -> Callable[[FunctionContext], Type]:
    def hook(ctx: FunctionContext) -> Type:
        return _refine_constructor_return(
            ctx, ctx.default_return_type, ctx.args, ctx.callee_arg_names, spec
        )

    return hook


def _make_method_hook(
    spec: ConstructorSpec,
) -> Callable[[MethodContext], Type]:
    def hook(ctx: MethodContext) -> Type:
        return _refine_constructor_return(
            ctx, ctx.default_return_type, ctx.args, ctx.callee_arg_names, spec
        )

    return hook


def _refine_constructor_return(
    ctx: FunctionContext | MethodContext,
    default_return: Type,
    args: list[list[Expression]],
    callee_arg_names: list[str | None],
    spec: ConstructorSpec,
) -> Type:
    """Refine an ``ndarray[tuple[int, ...], dtype[Any]]`` return into a more
    specific type when the shape and/or dtype call args are literals.

    ``spec`` describes where to look for the shape and dtype call expressions
    (positional index vs. keyword name).
    """
    if not args:
        return default_return

    ret = get_proper_type(default_return)
    # if not isinstance(ret, Instance) or ret.type.fullname != "numpy.ndarray":
    #     return default_return
    # if len(ret.args) < 2:
    #     return default_return

    int_instance = ctx.api.named_generic_type("builtins.int", [])
    fallback = ctx.api.named_generic_type("builtins.tuple", [int_instance])

    shape_expr = _find_shape_arg(args, callee_arg_names, spec.shape_arg)
    literal_dims = (
        _literal_dims_from_expr(shape_expr) if shape_expr is not None else None
    )

    new_shape: Type | None = None
    if literal_dims is not None:
        # should be shape_expr
        shape_type = get_proper_type(ret.args[0])
        if isinstance(shape_type, TupleType) and len(shape_type.items) == len(
            literal_dims
        ):
            new_items: list[Type] = []
            for existing, dim in zip(shape_type.items, literal_dims):
                if dim is None:
                    new_items.append(existing)
                else:
                    new_items.append(LiteralType(dim, fallback=int_instance))
            new_shape = TupleType(new_items, fallback=fallback)
        else:
            new_items = [
                LiteralType(d, fallback=int_instance) if d is not None else int_instance
                for d in literal_dims
            ]
            new_shape = TupleType(new_items, fallback=fallback)

    # Refine dtype when the call passed an explicit numpy generic class.
    new_dtype: Type | None = None
    if spec.dtype_arg is not None:
        dtype_expr = _find_named_arg(args, callee_arg_names, spec.dtype_arg)
        if dtype_expr is not None:
            dtype_instance = _dtype_instance_from_expr(dtype_expr, ctx)
            new_dtype = ctx.api.named_generic_type("numpy.dtype", [dtype_instance])

    if new_shape is None and new_dtype is None:
        return default_return

    new_args = list(ret.args)
    if new_shape is not None:
        new_args[0] = new_shape
    if new_dtype is not None:
        new_args[1] = new_dtype
    return ret.copy_modified(args=new_args)


def _find_shape_arg(
    args: list[list[Expression]],
    callee_arg_names: list[str | None],
    shape_arg: int | str,
) -> Expression | None:
    """Resolve a shape argument by positional index or keyword name."""
    if isinstance(shape_arg, int):
        if 0 <= shape_arg < len(args) and args[shape_arg]:
            return args[shape_arg][0]
        return None
    return _find_named_arg(args, callee_arg_names, shape_arg)


def _find_named_arg(
    args: list[list[Expression]],
    callee_arg_names: list[str | None],
    name: str,
) -> Expression | None:
    """Return the call expression bound to ``name``, or ``None`` if absent."""
    for i, candidate in enumerate(callee_arg_names):
        if candidate == name and i < len(args) and args[i]:
            return args[i][0]
    return None


def _dtype_instance_from_expr(
    expr: Expression, ctx: FunctionContext | MethodContext
) -> Type | None:
    """If ``expr`` is a class reference to a numpy generic, return its Instance."""
    from mypy.types import CallableType, TypeType

    expr_type = get_proper_type(ctx.api.get_expression_type(expr))

    # normal actual type
    if isinstance(expr_type, TypeType):
        inner = get_proper_type(expr_type.item)
        if isinstance(inner, Instance) and _is_numpy_generic(inner.type):
            return inner

    # (yes the if's purposely fall through, yes it's ugly, yes PRs welcome.)
    if isinstance(expr_type, CallableType):
        expr_type = get_proper_type(expr_type.ret_type)

    if isinstance(expr_type, Instance):
        # e.g. a dtype() instance: dtype=np.dtype(np.uint8)
        if expr_type.type.fullname == "numpy.dtype" and expr_type.args:
            expr_type = get_proper_type(expr_type.args[0])

        if expr_type.type.fullname in BUILTIN_TO_NUMPY:
            return ctx.api.named_type(BUILTIN_TO_NUMPY[expr_type.type.fullname])

    return expr_type


def _literal_dims_from_expr(expr: Expression) -> list[int | None] | None:
    """
    If ``expr`` is a literal tuple/list of integers, return them.

    Returns ``None`` for non-literal shapes, mixed types, or scalar-int shapes
    (the 1-D form ``np.empty(5)``).
    """
    if isinstance(expr, (TupleExpr, ListExpr)):
        dims: list[int | None] = []
        for item in expr.items:
            value = _int_literal_value(item)
            dims.append(value)  # may be None for a non-literal int element
        return dims
    if isinstance(expr, IntExpr):
        # 1-D constructor: np.empty(5) -> tuple[Literal[5]]
        return [expr.value]
    if (
        isinstance(expr, UnaryExpr)
        and expr.op == "-"
        and isinstance(expr.expr, IntExpr)
    ):
        return [-expr.expr.value]
    return None


def _int_literal_value(expr: Expression) -> int | None:
    if isinstance(expr, IntExpr):
        return expr.value
    if (
        isinstance(expr, UnaryExpr)
        and expr.op == "-"
        and isinstance(expr.expr, IntExpr)
    ):
        return -expr.expr.value
    return None
