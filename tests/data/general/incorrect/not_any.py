"""
The non-mypy checking isn't *so* generic that literally anything can be passed
"""

from numpydantic import NDArray, Shape


def func(x: NDArray[Shape["1, 2"]]) -> None:
    print(x)


func("hey")
