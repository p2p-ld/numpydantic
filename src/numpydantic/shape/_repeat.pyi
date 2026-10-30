from typing import TypeAlias

from typing_extensions import TypeVar

# The default makes bare Repeat tuple[int, ...]
_Size = TypeVar("_Size", bound=int, default=int)
Repeat: TypeAlias = tuple[_Size, ...]
