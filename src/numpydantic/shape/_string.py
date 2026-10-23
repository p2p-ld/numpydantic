import re
import string
import sys
import warnings
from typing import Annotated, Any, Literal, TypeAlias, get_args, get_origin

from numpydantic.exceptions import InvalidShapeExpressionError

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
_REGEX_REPEAT = rf"(({_REGEX_DIMENSION_SIZE}|{_REGEX_DIMENSION_RANGE})\.\.\.\s*)"
_REGEX_SHAPE_EXPRESSION_REPEAT = (
    rf"^(({_REGEX_DIMENSIONS}{_REGEX_SEPARATOR})?{_REGEX_REPEAT})$"
)

_RE_DIMENSION_COMPILED = re.compile(
    rf"(?P<dimension>{_REGEX_DIMENSION})(?P<label>\s+{_REGEX_LABEL})*"
)
_RE_SIZE_COMPILED = re.compile(_REGEX_DIMENSION_SIZE)


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


def _handle_ellipsis(shape: tuple, target: list[str]) -> list[str]:
    # Let the ellipsis allows for any number of dimensions by replacing the
    # ellipsis with the dimension size repeated the number of times that
    # corresponds to the shape of the instance.
    if isinstance(target, tuple):
        target = list(target)
    if target[-1] == "...":
        dim_to_repeat = "*"
    elif target[-1].endswith("..."):
        dim_to_repeat = target[-1].removesuffix("...")
    else:
        return target

    target = target[0:-1]
    if len(shape) > len(target):
        difference = len(shape) - len(target)
        target += difference * [dim_to_repeat]
    return target


def _is_range(target_dim: str) -> bool:
    """Whether the dimension is a range (literally whether it includes a hyphen)"""
    return "-" in target_dim and len(target_dim.split("-")) == 2


def _check_range(dim: str, target_dim: str) -> bool:
    """check whether the given dimension is within the target_dim range"""
    dim = int(dim)

    range_min, range_max = target_dim.split("-")
    if _is_wildcard(range_min):
        return dim <= int(range_max)
    elif _is_wildcard(range_max):
        return dim >= int(range_min)
    else:
        return int(range_min) <= dim <= int(range_max)


def _is_wildcard(dim: str) -> bool:
    """
    CHANGES FROM NPTYPING:
    - added '*-*' range, which is a wildcard
    - treat the raw `int` class as a wildcard, as it is in numpy types
    """
    # Return whether dim is a wildcard (i.e. the character that takes any
    # dimension size).
    return dim == "*" or dim == "*-*" or dim is int


def _is_assignable_var(dim: str, target_dim: str, variables: dict[str, str]) -> bool:
    # Return whether target_dim is a variable and can be assigned with dim.
    return _is_variable(target_dim) and _can_assign_variable(dim, target_dim, variables)


def _is_variable(dim: str) -> bool:
    # Return whether dim is a variable.
    return dim[0] in string.ascii_uppercase


def _can_assign_variable(dim: str, target_dim: str, variables: dict[str, str]) -> bool:
    # Check and assign a variable.
    assignable = variables.get(target_dim) in (None, dim)
    variables[target_dim] = dim
    return assignable


def normalize_string_shape_expression(shape_expression: StrShapeExpression) -> tuple:
    """
    Normalize the given shape expression, e.g. by removing whitespaces, making
    similar expressions look the same.
    :param shape_expression: the shape expression that is to be normalized.
    :return: a normalized shape expression.
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
    # rm quotes
    shape_expression = shape_expression.replace("'", "").replace('"', "")

    validate_string_shape_expression(shape_expression)

    str_parts = shape_expression.split(",")
    tuple_parts = tuple(_str_dim_to_tuple(part) for part in str_parts)

    return tuple_parts


def _str_dim_to_tuple(part: str) -> type:
    """Convert a single dimension from the string form to its canonical tuple form"""
    from numpydantic.shape import AnyDims, Range

    match = _RE_DIMENSION_COMPILED.fullmatch(part.strip())
    if not match:
        if part.strip() == "...":
            return next(iter(AnyDims))
        raise InvalidShapeExpressionError(
            f"{part} is not a valid dimension expression!"
        )
    dim = match.groupdict()["dimension"]
    if _is_range(dim):
        low, high = [int if d == "*" else Literal[int(d)] for d in dim.split("-")]
        inner = Range[low, high]
    elif _is_wildcard(dim):
        inner = int
    elif _RE_SIZE_COMPILED.fullmatch(dim):
        inner = Literal[int(dim)]
    else:
        raise InvalidShapeExpressionError(
            f"{part} is not a valid dimension expression!"
        )

    if label := match.groupdict().get("label"):
        return Annotated[inner, label.strip()]
    else:
        return inner


def validate_string_shape_expression(
    shape_expression: StrShapeExpression | Any,
) -> str:
    """
    CHANGES FROM NPTYPING:
    - Allow ranges
    - Allow specifying as a tuple
    """
    patterns = [_REGEX_SHAPE_EXPRESSION]

    if shape_expression is not Any and not any(
        re.match(pattern, shape_expression) for pattern in patterns
    ):
        raise InvalidShapeExpressionError(
            f"'{shape_expression}' is not a valid shape expression."
        )
    return shape_expression
