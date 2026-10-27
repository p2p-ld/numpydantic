"""
MIT License

Copyright (c) 2023 Ramon Hagenaars

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import string
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
)

from numpydantic.exceptions import InvalidShapeExpressionError
from numpydantic.shape._string import StrShapeExpression
from numpydantic.vendor.nptyping.typing_ import ShapeTuple

if TYPE_CHECKING:
    from nptyping.shape import Shape  # pragma: no cover


@lru_cache
def check_shape(shape: ShapeTuple, target: "Shape") -> bool:
    """
    Check whether the given shape corresponds to the given shape_expression.
    :param shape: the shape in question.
    :param target: the shape expression to which shape is tested.
    :return: True if the given shape corresponds to shape_expression.
    """
    target_shape = _handle_ellipsis(shape, target.prepared_args)
    return _check_dimensions_against_shape(shape, target_shape)


def validate_shape_expression(shape_expression: StrShapeExpression | Any) -> None:
    """
    Validate shape_expression and raise an InvalidShapeError if it is not
    considered valid.
    :param shape_expression: the shape expression to validate.
    :return: None.
    """
    shape_expression_no_quotes = shape_expression.replace("'", "").replace('"', "")
    if shape_expression is not Any and not re.match(
        _REGEX_SHAPE_EXPRESSION, shape_expression_no_quotes
    ):
        raise InvalidShapeExpressionError(
            f"'{shape_expression}' is not a valid shape expression."
        )


def get_dimensions(shape_expression: str) -> list[str]:
    """
    Find all "break downs" (the parts between brackets) in a shape expressions
    and replace them with mere dimension sizes.

    :param shape_expression: the shape expression that gets the break downs replaced.
    :return: a list of dimensions without break downs.
    """
    shape_expression_without_breakdowns = shape_expression
    for dim_breakdown in re.findall(
        r"(\[[^\]]+\])", shape_expression_without_breakdowns
    ):
        dim_size = len(dim_breakdown.split(","))
        shape_expression_without_breakdowns = (
            shape_expression_without_breakdowns.replace(dim_breakdown, str(dim_size))
        )
    return shape_expression_without_breakdowns.split(",")


def remove_labels(dimensions: list[str]) -> list[str]:
    """
    Remove all labels (words that start with a lowercase).

    :param dimensions: a list of dimensions.
    :return: a copy of the given list without labels.
    """
    return [re.sub(r"\b[a-z]\w*", "", dim).strip() for dim in dimensions]


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
        if dim != target_dim:
            return False
    return True


def _handle_ellipsis(shape: ShapeTuple, target: list[str]) -> list[str]:
    # Let the ellipsis allows for any number of dimensions by replacing the
    # ellipsis with the dimension size repeated the number of times that
    # corresponds to the shape of the instance.
    if target[-1] == "...":
        dim_to_repeat = target[-2]
        target = target[0:-1]
        if len(shape) > len(target):
            difference = len(shape) - len(target)
            target += difference * [dim_to_repeat]
    return target


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


def _is_wildcard(dim: str) -> bool:
    # Return whether dim is a wildcard (i.e. the character that takes any
    # dimension size).
    return dim == "*"


_REGEX_SEPARATOR = r"(\s*,\s*)"
_REGEX_DIMENSION_SIZE = r"(\s*[0-9]+\s*)"
_REGEX_VARIABLE = r"(\s*\b[A-Z]\w*\s*)"
_REGEX_LABEL = r"(\s*\b[a-z]\w*\s*)"
_REGEX_LABELS = rf"({_REGEX_LABEL}({_REGEX_SEPARATOR}{_REGEX_LABEL})*)"
_REGEX_WILDCARD = r"(\s*\*\s*)"
_REGEX_DIMENSION_BREAKDOWN = rf"(\s*\[{_REGEX_LABELS}\]\s*)"
_REGEX_DIMENSION = (
    rf"({_REGEX_DIMENSION_SIZE}"
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
