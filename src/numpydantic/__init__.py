# ruff: noqa: F401
# ruff: noqa: I001
# ruff: noqa: D104

from numpydantic.ndarray import NDArray
from numpydantic.validation.shape import Shape
from numpydantic.annotation import NDArraySchema

__all__ = ["NDArray", "NDArraySchema", "Shape"]
