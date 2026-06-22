"""Special handlers for pydantic models"""

import contextlib
from collections.abc import Callable
from functools import partial

from mypy.nodes import FuncDef
from mypy.plugin import ClassDefContext
from mypy.types import UnionType
from pydantic.mypy import PydanticModelTransformer, PydanticPluginConfig


def _pydantic_hook(
    ctx: ClassDefContext, config: PydanticPluginConfig, inputs: list[str]
) -> None:
    """
    Override how pydantic init methods are declared so that we allow
    extra input types in the __init__ method but *not* on the type annotation itself.
    """
    transformer = PydanticModelTransformer(ctx.cls, ctx.reason, ctx.api, config)
    transformer.transform()
    # add input types to the init method
    candidates = [
        item
        for item in ctx.cls.defs.body
        if isinstance(item, FuncDef) and item.name == "__init__"
    ]
    if not candidates:
        return

    init = candidates[0]
    _unionize_init(ctx, init, inputs)


def _unionize_init(ctx: ClassDefContext, init: FuncDef, inputs: list[str]) -> None:
    # unionize ndarray types to accept nonstandard inputs
    input_types = None
    for i, arg in enumerate(init.arguments):
        if (
            not arg.type_annotation
            or not hasattr(arg.type_annotation, "type")
            or not arg.type_annotation.type.metadata.get("numpydantic")
        ):
            continue
        if input_types is None:
            input_types = []
            for name in inputs:
                with contextlib.suppress(AssertionError):
                    input_types.append(ctx.api.named_type(name, []))

        union = UnionType([arg.type_annotation, *input_types])
        arg.type_annotation = union
        init.type.arg_types[i] = union


def _make_pydantic_hook(
    config: PydanticPluginConfig, inputs: list[str]
) -> Callable[[ClassDefContext], None]:
    return partial(_pydantic_hook, config=config, inputs=inputs)
