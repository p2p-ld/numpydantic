"""
Helper functions for validation
"""

from numpydantic.shape import validate_shape
from numpydantic.validation.dtype import validate_dtype

__all__ = [
    "validate_dtype",
    "validate_shape",
]
