"""Static type stub for :class:`numpydantic.validation.shape.Shape`.

Mirrors the trick used by :mod:`numpydantic.vendor.nptyping.shape` — for
mypy ``Shape`` is cast to ``Literal``, for pyright it's a permissive class
whose subscript always returns ``Any``.
"""

from typing import Any, Literal, cast

Shape = cast(Literal, Shape)  # type: ignore[has-type,misc,valid-type]

class Shape:  # type: ignore[no-redef]
    def __class_getitem__(cls, item: Any, /) -> Any: ...
    def __new__(cls, *args: Any) -> Any: ...
