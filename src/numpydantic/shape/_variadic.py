import sys
from typing import Any, Unpack, get_args, get_origin


def _unpacked_tuple_args(item: Any) -> tuple | None:
    """The items of an unpacked tuple, written as *tuple[...] or Unpack[tuple[...]]"""
    if sys.version_info < (3, 12):
        # TODO: remove when dropping python 3.11, where Unpack[tuple[...]]
        # lacks this attribute, but *tuple[...] has it
        if get_origin(item) is Unpack:
            item = next(iter(get_args(item)[0]))
    return getattr(item, "__typing_unpacked_tuple_args__", None)


def _variadic_item(item: Any) -> Any | None:
    """The repeated item of an unpacked tuple[X, ...], or None"""
    args = _unpacked_tuple_args(item)
    if args is not None and len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return None


def _split_variadic(args: tuple) -> tuple[tuple, Any | None]:
    """Split shape args into the fixed dimensions and the repeated trailing dimension"""
    if len(args) == 2 and args[1] is Ellipsis:
        return (), args[0]
    if args and (item := _variadic_item(args[-1])) is not None:
        return args[:-1], item
    return args, None


def _is_any_shape(args: tuple) -> bool:
    if len(args) == 1 and args[0] is Any:
        return True
    fixed, tail = _split_variadic(args)
    # (int, *Repeat) is how "*, ..." is parsed, which has always meant any shape,
    # including 0-d arrays
    return tail is int and fixed in ((), (int,))
