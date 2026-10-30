# Shape is literally tuple to type checkers, so that it is a valid type expression
# with the same meaning, including Shape[X, ...]. A renamed import is only
# re-exported from a stub when it's listed in __all__.
from builtins import tuple as Shape

__all__ = [
    "Shape",
]
