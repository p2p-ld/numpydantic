# ruff: noqa: E402
# ruff: noqa: F401
# ruff: noqa: I001
from numpydantic.monkeypatch import apply_patches

apply_patches()

from numpydantic.ndarray import NDArray

# convenience imports for typing - finish this!
from typing import Any

from nptyping import Float, Int, Number, Shape, UInt8
