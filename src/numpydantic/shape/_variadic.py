import sys
from typing import Any, get_origin

if sys.version_info < (3, 11):
    from typing_extensions import Unpack
else:
    from typing import Unpack


def _is_unpacked(item: Any) -> bool:
    return getattr(item, "__unpacked__", False) or get_origin(item) is Unpack


def _variadic_item(item: Any) -> Any | None:
    """The repeated item of an unpacked tuple[X, ...], or None"""
    # typing's own attribute for the items of an unpacked tuple, as * or Unpack
    args = getattr(item, "__typing_unpacked_tuple_args__", None)
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
    # (int, *AnyDims) is how "*, ..." is parsed, which has always meant any shape,
    # including 0-d arrays
    return tail is int and fixed in ((), (int,))
