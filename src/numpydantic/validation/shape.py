"""
Declaration and validation functions for array shapes.

Mostly a mildly modified version of nptyping's
:func:`npytping.shape_expression.validate_shape`
and its internals to allow for extended syntax, including ranges of shapes.

Modifications from nptyping:

- **"..."** - In nptyping, ``'...'`` means "any number of dimensions with the same shape
  as the last dimension. ie ``Shape[2, ...]`` means "any number of 2-length
  dimensions. Here ``'...'`` always means "any number of any-shape dimensions"
- **Ranges** - (inclusive) shape ranges are allowed. eg. to specify an array
  where the first dimension can be 2, 3, or 4 length::

     Shape["2-4, ..."]

  To specify a range with an unbounded min or max, use wildcards, eg. for
  an array with the first dimension at least length 2, and the second dimension
  at most length 5 (both inclusive)::

      Shape["2-*, *-5"]

"""

import re
import string
import sys
from abc import ABC
from functools import lru_cache
from typing import Any, Generic, Literal, get_args, get_origin

from numpydantic.vendor.nptyping.base_meta_classes import ContainerMeta
from numpydantic.vendor.nptyping.error import InvalidShapeError
from numpydantic.vendor.nptyping.nptyping_type import NPTypingType
from numpydantic.vendor.nptyping.shape_expression import (
    get_dimensions,
    normalize_shape_expression,
    remove_labels,
)
from numpydantic.vendor.nptyping.typing_ import ShapeExpression, ShapeTuple

if sys.version_info < (3, 11):
    from typing_extensions import TypeVarTuple, Unpack
else:
    from typing import TypeVarTuple, Unpack

_TV = TypeVarTuple("_TV")


class ShapeMeta(ContainerMeta["Shape"], implementation="Shape"):
    """
    Metaclass that is coupled to nptyping.Shape.

    Overridden from nptyping to use local shape validation function
    """

    def _validate_expression(cls, item: str) -> str:
        return validate_shape_expression(item)

    def _normalize_expression(cls, item: str) -> str:
        return normalize_shape_expression(item)

    def _get_additional_values(cls, item: Any) -> dict[str, Any]:
        if isinstance(item, tuple):
            item = _normalize_tuple(item)
        dim_strings = get_dimensions(item)
        dim_string_without_labels = remove_labels(dim_strings)
        return {"prepared_args": dim_string_without_labels}


class Shape(NPTypingType, ABC, Generic[Unpack[_TV]], metaclass=ShapeMeta):
    """
    A container for shape expressions that describe the shape of an multi
    dimensional array.

    Simple example:

    >>> Shape['2, 2']
    Shape['2, 2']

    A Shape can be compared to a typing.Literal. You can use Literals in
    NDArray as well.

    >>> from typing import Literal

    >>> Shape['2, 2'] == Literal['2, 2']
    True

    A Shape can be constructed by calling for type checker compatibility

    >>> Shape['2, 2'] == Shape('2, 2')

    And its arguments can be pased as ``args``, with ints and strings as appropriate

    >>> Shape(2, 2, "...") == Shape("2, 2, ...")

    """

    def __new__(cls, *args: str | int) -> type["Shape"]:
        """Create a new Shape as a callable"""
        return Shape[args]

    __args__ = ("*, ...",)
    prepared_args = ("*", "...")


def validate_shape_expression(
    shape_expression: ShapeExpression | tuple[str, ...] | Any,
) -> str:
    """
    CHANGES FROM NPTYPING:
    - Allow ranges
    - Allow specifying as a tuple
    """
    if isinstance(shape_expression, tuple):
        shape_expression = _normalize_tuple(shape_expression)
    shape_expression_no_quotes = shape_expression.replace("'", "").replace('"', "")
    if shape_expression is not Any and not re.match(
        _REGEX_SHAPE_EXPRESSION, shape_expression_no_quotes
    ):
        raise InvalidShapeError(
            f"'{shape_expression}' is not a valid shape expression."
        )
    return shape_expression


def _normalize_tuple(expression: tuple) -> str:
    """
    Normalize tuple args to a nptyping string.

    FIXME: This whole 'convert everything down to a string' thing has gotta go
    """
    terms = []
    for t in expression:
        if t is int:
            t = "*"
        elif t is Ellipsis:
            t = "..."
        elif get_origin(t) is Literal:
            t = get_args(t)[0]

        terms.append(t)
    return ", ".join(str(term) for term in terms)


@lru_cache
def validate_shape(shape: ShapeTuple, target: "Shape") -> bool:
    """
    Check whether the given shape corresponds to the given shape_expression.
    :param shape: the shape in question.
    :param target: the shape expression to which shape is tested.
    :return: True if the given shape corresponds to shape_expression.
    """
    target_shape = _handle_ellipsis(shape, target.prepared_args)
    return _check_dimensions_against_shape(shape, target_shape)


def _check_dimensions_against_shape(shape: ShapeTuple, target: list[str]) -> bool:
    # Walk through the shape and test them against the given target,
    # taking into consideration variables, wildcards, etc.

    if len(shape) != len(target):
        return False
    shape_as_strings = (str(dim) for dim in shape)
    variables: dict[str, str] = {}
    for dim, target_dim in zip(shape_as_strings, target):
        if _is_wildcard(target_dim) or _is_assignable_var(dim, target_dim, variables):
            continue
        if _is_range(target_dim) and _check_range(dim, target_dim):
            continue
        if dim != target_dim:
            return False
    return True


def _handle_ellipsis(shape: ShapeTuple, target: list[str]) -> list[str]:
    # Let the ellipsis allows for any number of dimensions by replacing the
    # ellipsis with the dimension size repeated the number of times that
    # corresponds to the shape of the instance.
    if isinstance(target, tuple):
        target = list(target)
    if target[-1] == "...":
        dim_to_repeat = "*"
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


# CHANGES FROM NPTYPING: Allow ranges
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

# --------------------------------------------------
# Below - unchanged from nptyping
# --------------------------------------------------


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
