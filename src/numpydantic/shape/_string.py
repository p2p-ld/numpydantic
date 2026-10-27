import re
import sys
import warnings
from typing import Annotated, Literal, TypeAlias, get_args, get_origin

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._range import Range

StrShapeExpression: TypeAlias = str

_REGEX_SEPARATOR = r"(\s*,\s*)"
_REGEX_DIMENSION_SIZE = r"(\s*[0-9]+\s*)"
_REGEX_DIMENSION_RANGE = r"(\s*[0-9\*]+-[0-9\*]+\s*)"
_REGEX_VARIABLE = r"(\s*\b[A-Z]\w*\s*)"
_REGEX_LABEL = r"(\s*\b[a-z]\w*\s*)"
_REGEX_LABELS = rf"({_REGEX_LABEL}({_REGEX_SEPARATOR}{_REGEX_LABEL})*)"
_REGEX_WILDCARD = r"(\s*\*\s*)"
_REGEX_DIMENSION_BREAKDOWN = rf"(\s*\[{_REGEX_LABELS}\]\s*)"
_REGEX_DIMENSION = (
    rf"({_REGEX_DIMENSION_SIZE}"
    rf"|{_REGEX_DIMENSION_RANGE}"
    rf"|{_REGEX_VARIABLE}"
    rf"|{_REGEX_WILDCARD}"
    rf"|{_REGEX_DIMENSION_BREAKDOWN})"
)
_REGEX_DIMENSION_WITH_LABEL = rf"({_REGEX_DIMENSION}(\s+{_REGEX_LABEL})*)"
_REGEX_DIMENSIONS = (
    rf"{_REGEX_DIMENSION_WITH_LABEL}({_REGEX_SEPARATOR}{_REGEX_DIMENSION_WITH_LABEL})*"
)
_REGEX_DIMENSIONS_ELLIPSIS = rf"({_REGEX_DIMENSIONS}{_REGEX_SEPARATOR}\.\.\.\s*)"
_REGEX_SHAPE_EXPRESSION = rf"^({_REGEX_DIMENSIONS}|{_REGEX_DIMENSIONS_ELLIPSIS})$"

_RE_DIMENSION_COMPILED = re.compile(
    rf"(?P<dimension>{_REGEX_DIMENSION})(?P<label>\s+{_REGEX_LABEL})*"
)
_RE_SIZE_COMPILED = re.compile(_REGEX_DIMENSION_SIZE)
_RE_VARIABLE_COMPILED = re.compile(_REGEX_VARIABLE)


def _string_terms(item: str | tuple) -> list[str]:
    """The parts of a shape subscript that are written as strings"""
    terms = (item,) if not isinstance(item, tuple) else item
    strings = []
    for term in terms:
        if get_origin(term) is Literal:
            term = get_args(term)[0]
        if isinstance(term, str):
            strings.append(term)
    return strings


def _warn_string_expression(item: str | tuple) -> None:
    strings = _string_terms(item)
    if not strings:
        return

    message = (
        f"String shape expressions like {item!r} are deprecated, "
        "use type expressions instead, e.g. "
        "Shape[L[3], int, Range[L[2], L[4]], *AnyDims]"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=_user_stacklevel())
    if any("..." in s for s in strings):
        message = (
            "Treating '...' in a string as meaning any number of dimensions "
            "of any size is now written as *AnyDims. "
            "The old behavior is deprecated in 2.0 and will be removed in 3.0."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=_user_stacklevel())


def _user_stacklevel() -> int:
    """
    Stack level of the first frame outside of numpydantic,
    for use in the function that calls ``warnings.warn``.
    """
    frame = sys._getframe(2)
    level = 2
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        "numpydantic"
    ):
        frame = frame.f_back
        level += 1
    return level


def normalize_string_shape_expression(shape_expression: StrShapeExpression) -> tuple:
    """
    Parse a string shape expression into the equivalent tuple form,
    one item per dimension.
    """
    shape_expression = shape_expression.replace("'", "").replace('"', "")
    # Replace whitespaces right before labels with $.
    shape_expression = re.sub(rf"\s*{_REGEX_LABEL}", r"$\1", shape_expression)
    # Let all commas be followed by a $.
    shape_expression = shape_expression.replace(",", ",$")
    # Remove all whitespaces left.
    shape_expression = re.sub(r"\s*", "", shape_expression)
    # Remove $ right after a bracket.
    shape_expression = re.sub(r"\[\$+", "[", shape_expression)
    # Replace $ with a single space.
    shape_expression = re.sub(r"\$+", " ", shape_expression)

    validate_string_shape_expression(shape_expression)
    return tuple(_str_dim_to_tuple(part) for part in shape_expression.split(","))


def validate_string_shape_expression(shape_expression: str) -> str:
    """Raise if a string shape expression isn't well formed"""
    if not re.match(_REGEX_SHAPE_EXPRESSION, shape_expression):
        raise InvalidShapeExpressionError(
            f"'{shape_expression}' is not a valid shape expression."
        )
    return shape_expression


def _str_dim_to_tuple(part: str) -> type:
    """Convert a single dimension from the string form to its canonical tuple form"""
    if "," in part:
        raise InvalidShapeExpressionError(
            f"A string for a single dimension can't contain a comma, got {part!r}"
        )
    match = _RE_DIMENSION_COMPILED.fullmatch(part.strip())
    if not match:
        if part.strip() == "...":
            # *AnyDims
            return next(iter(tuple[int, ...]))
        raise InvalidShapeExpressionError(
            f"{part} is not a valid dimension expression!"
        )
    dim = match.groupdict()["dimension"]
    if _is_range(dim):
        low, high = [int if d == "*" else Literal[int(d)] for d in dim.split("-")]
        inner = Range[low, high].annotation
    elif dim == "*":
        inner = int
    elif _RE_SIZE_COMPILED.fullmatch(dim):
        inner = Literal[int(dim)]
    elif _RE_VARIABLE_COMPILED.fullmatch(dim):
        raise InvalidShapeExpressionError(
            f"Shape variables like {dim!r} are not supported"
        )
    else:
        raise InvalidShapeExpressionError(
            f"{part} is not a valid dimension expression!"
        )

    if label := match.groupdict().get("label"):
        return Annotated[inner, label.strip()]
    else:
        return inner


def _is_range(dim: str) -> bool:
    """Whether the dimension is a range (literally whether it includes a hyphen)"""
    return "-" in dim and len(dim.split("-")) == 2
