"""
Helper functions for validation
"""

from numpydantic.validation.dtype import validate_dtype
from numpydantic.validation.shape import validate_shape

__all__ = [
    "validate_dtype",
    "validate_shape",
]
