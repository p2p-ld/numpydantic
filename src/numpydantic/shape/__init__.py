"""
Array shapes, written like tuple types: see :mod:`numpydantic.shape.shape`
"""

from numpydantic.shape._helpers import transposed, unliteral, unshape
from numpydantic.shape._range import Range
from numpydantic.shape._repeat import Repeat
from numpydantic.shape._validate import validate_shape
from numpydantic.shape.shape import Shape

__all__ = [
    "Range",
    "Repeat",
    "Shape",
    "transposed",
    "unliteral",
    "unshape",
    "validate_shape",
]
