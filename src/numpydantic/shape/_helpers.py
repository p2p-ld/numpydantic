# Some type variables below only appear in a parameter, which pyright warns about.
# pyright: reportInvalidTypeVarUse=false
from typing import Any, TypeVar, overload

import numpy as np

_DT = TypeVar("_DT", bound=np.dtype[Any])
_A = TypeVar("_A", bound=int)
_B = TypeVar("_B", bound=int)
_C = TypeVar("_C", bound=int)
_D = TypeVar("_D", bound=int)
_E = TypeVar("_E", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])


def unshape(array: np.ndarray[Any, _DT]) -> np.ndarray[Any, _DT]:
    """
    Forget an array's static shape, keeping its dtype.

    numpy often types sizes it can't know as plain ``int``,
    which can't be assigned to a ``Literal`` shape.
    The returned array takes its shape from whatever it's assigned to next::

        x: NDArray[Shape[L[3], L[3]], np.float64] = unshape(np.ones((n, n)))

    This does nothing at runtime and checks nothing.
    To claim a specific shape, use :func:`typing.cast`.
    """
    return array


# This has to come first, as in transposed below.
@overload
def unliteral(
    array: np.ndarray[tuple[int, int, int, int, int, int, *tuple[int, ...]], _DT],
) -> np.ndarray[tuple[Any, ...], _DT]: ...
@overload
def unliteral(array: np.ndarray[tuple[()], _DT]) -> np.ndarray[tuple[()], _DT]: ...
@overload
def unliteral(array: np.ndarray[tuple[_A], _DT]) -> np.ndarray[tuple[int], _DT]: ...
@overload
def unliteral(
    array: np.ndarray[tuple[_A, _B], _DT],
) -> np.ndarray[tuple[int, int], _DT]: ...
@overload
def unliteral(
    array: np.ndarray[tuple[_A, _B, _C], _DT],
) -> np.ndarray[tuple[int, int, int], _DT]: ...
@overload
def unliteral(
    array: np.ndarray[tuple[_A, _B, _C, _D], _DT],
) -> np.ndarray[tuple[int, int, int, int], _DT]: ...
@overload
def unliteral(
    array: np.ndarray[tuple[_A, _B, _C, _D, _E], _DT],
) -> np.ndarray[tuple[int, int, int, int, int], _DT]: ...


# The overloads above and below match the shape with type variables
# so that mypy doesn't treat a call with an Any dtype as matching ambiguously.
@overload
def unliteral(array: np.ndarray[_S, _DT]) -> np.ndarray[tuple[int, ...], _DT]: ...
def unliteral(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Forget an array's static sizes, keeping its number of dimensions and dtype.

    numpy types many functions that change sizes, like :func:`numpy.pad`,
    as returning the shape they were given.
    Use this when the sizes are wrong or unknown but the number of dimensions isn't::

        a: NDArray[Shape[L[3], L[4]], np.float64]
        b: NDArray[Shape[int, int], np.float64] = unliteral(np.pad(a, 1))

    The result can't be used where literal sizes are expected:
    use :func:`.unshape` or :func:`typing.cast` to give it literal sizes again.

    This does nothing at runtime.

    Shapes with up to 5 dimensions keep their number of dimensions.
    Unknown shapes and shapes with more dimensions come back as ``tuple[Any, ...]``,
    and shapes with an unknown number of dimensions as ``tuple[int, ...]``.
    """
    return array


# This has to come first: a shape of unknown length, tuple[Any, ...],
# would otherwise match the first fixed-length overload below.
@overload
def transposed(
    array: np.ndarray[tuple[int, int, int, int, int, int, *tuple[int, ...]], _DT],
) -> np.ndarray[tuple[Any, ...], _DT]: ...
@overload
def transposed(array: np.ndarray[tuple[()], _DT]) -> np.ndarray[tuple[()], _DT]: ...
@overload
def transposed(array: np.ndarray[tuple[_A], _DT]) -> np.ndarray[tuple[_A], _DT]: ...
@overload
def transposed(
    array: np.ndarray[tuple[_A, _B], _DT],
) -> np.ndarray[tuple[_B, _A], _DT]: ...
@overload
def transposed(
    array: np.ndarray[tuple[_A, _B, _C], _DT],
) -> np.ndarray[tuple[_C, _B, _A], _DT]: ...
@overload
def transposed(
    array: np.ndarray[tuple[_A, _B, _C, _D], _DT],
) -> np.ndarray[tuple[_D, _C, _B, _A], _DT]: ...
@overload
def transposed(
    array: np.ndarray[tuple[_A, _B, _C, _D, _E], _DT],
) -> np.ndarray[tuple[_E, _D, _C, _B, _A], _DT]: ...


# mypy falls back to this when an assigned annotation doesn't match a shape above,
# so it has to give a shape that annotation refuses.
# The shape is a type variable for the same reason as in unliteral.
@overload
def transposed(array: np.ndarray[_S, _DT]) -> np.ndarray[tuple[int, ...], _DT]: ...
def transposed(array: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Reverse the static shape of an array that numpy has transposed.

    numpy types ``.T``, ``.transpose()`` and :func:`numpy.transpose` as returning
    the same shape they were given, so a ``(3, 4)`` array stays ``(3, 4)``
    to type checkers. Wrap the result to reverse its shape::

        a: NDArray[Shape[L[3], L[4]], np.float64]
        b: NDArray[Shape[L[4], L[3]], np.float64] = transposed(a.T)

    This does nothing at runtime: it doesn't transpose the array itself.

    Shapes with up to 5 dimensions are reversed.
    Unknown shapes and shapes with more dimensions come back as ``tuple[Any, ...]``,
    and shapes with an unknown number of ``int`` sizes as ``tuple[int, ...]``.
    """
    return array
