"""
Per-interface static-typing for the mypy plugin

A backend interface may attach an :class:`InterfaceTyping` subclass to its
:attr:`Interface.typing` class variable to opt into:

- Mypy plugin constructor inference (so ``da.zeros((3, 3), dtype=np.uint8)``
  refines to a literal-tuple shape and concrete dtype in the same way numpy
  constructors do).
- Mypy test file generation (the test generator uses the typing class to
  emit literal constructor expressions for each combinatorial test case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ConstructorSpec:
    """
    How a single array-constructor call exposes shape and dtype.
    """

    fullname: str
    """
    Mypy-visible fully qualified name of the function or method,
    e.g. ``"dask.array.zeros"`` or
    ``"numpy._core.multiarray._ConstructorEmpty.__call__"``.
    """
    shape_arg: int | str = 0
    """
    Where the shape argument lives in the call. 
    An ``int`` is a positional index; 
    a ``str`` is a keyword name.
    """
    dtype_arg: str | None = "dtype"
    """Keyword name for the dtype argument, if any"""
    is_method: bool = False
    """
    ``True`` if mypy sees this constructor as a method (``get_method_hook``), 
    ``False`` for a free function (``get_function_hook``).
    """


class InterfaceTyping:
    """
    Optional static-typing companion class for an :class:`Interface`,
    for use with the mypy plugin.
    """

    constructors: ClassVar[tuple[ConstructorSpec, ...]] = ()
    """Constructor calls whose return type the mypy plugin should refine."""

    @classmethod
    def emit_imports(cls) -> list[str]:
        """Import lines required by :meth:`emit_constructor_source`.

        Returned as a list of statement strings, e.g.
        ``["import numpy as np", "import dask.array as da"]``.
        """
        return []

    @classmethod
    def emit_constructor_source(cls, shape: tuple[int, ...], dtype: str) -> str | None:
        """Source-text for a constructor call producing the given array.

        Returns a Python expression string (no trailing newline), e.g.
        ``"np.zeros((3, 3), dtype=np.uint8)"``.

        Only needs to render one of the constructors that the interface supports -
        we assume if we detect one, they should all work the same way.
        """
        return None
