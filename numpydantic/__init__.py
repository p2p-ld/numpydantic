# ruff: noqa: E402
# ruff: noqa: F401
from numpydantic.monkeypatch import apply_patches

apply_patches()

from numpydantic.ndarray import NDArray
