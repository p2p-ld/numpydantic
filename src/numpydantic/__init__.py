# ruff: noqa: F401
# ruff: noqa: I001
# ruff: noqa: D104

from numpydantic.ndarray import NDArray
from numpydantic.meta import update_ndarray_stub
from numpydantic.shape import Shape

update_ndarray_stub()

__all__ = ["NDArray", "Shape"]
