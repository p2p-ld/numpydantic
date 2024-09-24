"""
Exceptions used within numpydantic
"""


class InterfaceError(Exception):
    """Parent mixin class for errors raised by :class:`.Interface` subclasses"""


class DtypeError(ValueError, TypeError, InterfaceError):
    """Exception raised for invalid dtypes"""


class ShapeError(ValueError, InterfaceError):
    """Exception raise for invalid shapes"""


class MatchError(ValueError, InterfaceError):
    """Exception for errors raised during :class:`.Interface.match`-ing"""


class NoMatchError(MatchError):
    """No match was found by :class:`.Interface.match`"""


class TooManyMatchesError(MatchError):
    """Too many matches found by :class:`.Interface.match`"""


class MarkMismatchError(MatchError):
    """A serialized :class:`.InterfaceMark` doesn't match the receiving interface"""
