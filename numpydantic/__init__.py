# ruff: noqa: E402
# ruff: noqa: F401
from numpydantic.monkeypatch import apply_patches

apply_patches()

# convenience imports for typing - finish this!
from typing import Any

from nptyping import Float, Int, Number, Shape, UInt8

from numpydantic.ndarray import NDArray
