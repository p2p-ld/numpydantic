"""
The non-mypy checking isn't *so* generic that literally anything can be passed
"""

from typing import Literal as L

from numpydantic import NDArray, Shape


def func(x: NDArray[Shape[L[1], L[2]]]) -> None:
    print(x)


func("hey")  # E: incompatible type
