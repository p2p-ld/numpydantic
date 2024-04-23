"""
Exceptions used within numpydantic
"""


class DtypeError(TypeError):
    """Exception raised for invalid dtypes"""


class ShapeError(ValueError):
    """Exception raise for invalid shapes"""
