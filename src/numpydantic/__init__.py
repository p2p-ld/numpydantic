# ruff: noqa: E402
# ruff: noqa: F401
# ruff: noqa: I001
from numpydantic.monkeypatch import apply_patches

apply_patches()

from numpydantic.ndarray import NDArray

from numpydantic.meta import update_ndarray_stub

update_ndarray_stub()
