# Syntax

General form:

```python
from typing import Literal as L
from numpydantic import NDArray, Shape

field: NDArray[Shape[L[3], int], dtype]
```

## Dtype

Dtype checking is for the most part as simple as an `isinstance` check - 
the `dtype` attribute of the array is checked against the `dtype` provided in the
`NDArray` annotation. Both numpy and builtin python types can be used.

A tuple of types can also be passed:

```python
field: NDArray[Shape[L[2], L[3]], (np.int8, np.uint8)]
```

The {mod}`~numpydantic.dtype` module provides convenient access
and aliases to the common dtypes, but also provides "generic" dtypes like
{class}`~numpydantic.dtype.Float` that is a tuple of all subclasses of 
{class}`numpy.floating`. Numpy interprets `float` as being equivalent to 
{class}`numpy.float64`, and {class}`numpy.floating` is an abstract parent class, 
so "generic" tuple dtypes fill that narrow gap.

```{todo}
Future versions will support interfaces providing type maps for declaring
equality between dtypes that may be specific to that library but should be 
considered equivalent to numpy or other library's dtypes.
```

```{todo}
Future versions will also support declaring minimum or maximum precisions, 
so one might say "at least a 16-bit float" and also accept a 32-bit float.
```

## Shape

A shape is written like a tuple type, with one item per dimension,
the same way numpy annotates the shape of an array:

```python
import numpy as np
from typing import Any, TypeVar

_T_Shape = TypeVar("_T_Shape", bound=tuple[Any, ...], default=tuple)
_T_Dtype = TypeVar("_T_Dtype", bound=np.generic, default=Any)

NDArray = np.ndarray[_T_Shape, np.dtype[_T_Dtype]]
```

To type checkers, {class}`.Shape` *is* `tuple`, so `Shape[...]` and `tuple[...]`
are interchangeable, and both are valid type expressions
(see [Typechecker Integration](typecheckers.md)).
At runtime, `Shape[...]` returns the equivalent `tuple[...]`,
checking the dimensions and normalizing them into a consistent form.

The examples below use these imports:

```python
from typing import Annotated as A, Literal as L
from numpydantic import NDArray, Range, Repeat, Shape
```

### Exact sizes

A {class}`typing.Literal` is a dimension of exactly that size.

For a 2-dimensional, 3 x 4-shaped array:

```python
Shape[L[3], L[4]]
```

A single dimension is just one item:

```python
Shape[L[3]]
```

A `Literal` with several values is a dimension that can be any one of those sizes:

```python
# 3 or 5, then 2
Shape[L[3, 5], L[2]]
```

### Any size

`int` is a dimension of any size.

For a 2-dimensional, 3 x any-shaped array:

```python
Shape[L[3], int]
```

(shape-ranges)=
### Ranges

{class}`.Range` is a dimension whose size is within a range.
Ranges are **inclusive** on both ends.
A `Literal` sets a bound, and `int` is open-ended.

For an array whose...
- First dimension can be of length 2, 3, or 4
- Second dimension is 2 or greater
- Third dimension is 4 or less

```python
Shape[Range[L[2], L[4]], Range[L[2], int], Range[int, L[4]]]
```

Type checkers can't check ranges on their own, so to them a range is just an `int`.
The [mypy plugin](mypy-plugin) checks ranges, including that a range
fits within the range it is assigned to.

### Unions

A union of dimensions is a dimension that matches any one of them:

```python
# 3 or 5, the same as L[3, 5]
Shape[L[3] | L[5]]
# 1 to 2, or 6 to 7
Shape[Range[L[1], L[2]] | Range[L[6], L[7]]]
# exactly 3, or 6 to 7
Shape[L[3] | Range[L[6], L[7]]]
```

(shape-repeat)=
### Repeated Dimensions

As with `tuple[X, ...]`, `Shape[X, ...]` is any number of dimensions matching `X`:

```python
# any number of 2s, including none
Shape[L[2], ...]
```

This is only valid when a single dimension is declared within the shape.
When multiple dimensions are declared, to repeat the final dimension, unpack {class}`.Repeat`:

```python
from numpydantic import Repeat
# 1 dimension of length 1, then any number (including 0) of length-2 dimensions
Shape[L[1], *Repeat[L[2]]]
# A shorthand equivalent for
Shape[L[1], *tuple[L[2], ...]]

# any number of dimensions that are each 2 to 4 long
Shape[L[3], *Repeat[Range[L[2], L[4]]]]
# Any number of dimensions all of the same size
N = TypeVar("N", bound=int)
Shape[L[1], *Repeat[N]]
```

For any number of trailing dimensions of any shape:

```python
Shape[L[1], *Repeat]
# A shorthand equivalent for
Shape[L[1], *tuple[int, ...]]
```

Repeated dimensions must:

- be unpacked with `*`: `Shape[L[3], Repeat]` is an error
- be the last item in a shape: `Shape[*Repeat, L[3]]` is an error

### Labels

Dimensions can be given labels with {class}`typing.Annotated`.
They don't change the shape, and in future versions they will be
propagated to the generated JSON Schema.

```python
Shape[A[L[3], "x"], A[L[4], "y"], A[int, "z"]]
```

(shape-any)=
### Any shape

Use `Any` for an array of any shape:

```python
field: NDArray[Any, int]
```

If `dtype` is also `Any`, one can just use

```python
field: NDArray
```

### Zero-dimensional arrays

As with `tuple[()]`, an empty shape is a zero-dimensional array:

```python
Shape[()]
```

(shape-typevars)=
(shape-variables)=
### Shape typevars

```{admonition} Experimental!
:class: warning

Shape typevars are experimental, and how they're validated across the fields
of a model may change. pydantic doesn't give a type like `NDArray` a way to share
state between fields, so the sizes are kept in the model's `__dict__`,
which is a "terrible" way to do it, and only works because pydantic doesn't use
`__dict__` like how every other python class does. 
When pydantic's plugin system is stable, we may move to that instead.
```

Shapes can use a {class}`typing.TypeVar` for a dimension
(also referred to as a shape variable because saying "shape type var" out loud sucks really bad).
A shape typevar is a dimension of any size,
as long as it's the same size everywhere it's used.

A shape typevar has to be bound or constrained to sizes.
The simplest is bound to `int`, which can be any size:

`````{tab-set}
````{tab-item} Python >= 3.12
```python
# a square matrix
def trace[N: int](matrix: NDArray[Shape[N, N], np.float64]) -> float: ...

# any number of dimensions, all the same size
def fill[N: int](array: NDArray[Shape[N, ...], np.float64]) -> None: ...
```
````
````{tab-item} Python < 3.12
```python
N = TypeVar("N", bound=int)

# a square matrix
def trace(matrix: NDArray[Shape[N, N], np.float64]) -> float: ...

# any number of dimensions, all the same size
def fill(array: NDArray[Shape[N, ...], np.float64]) -> None: ...
```
````
`````

Like other type variables, a shape typevar belongs to the function or class that uses it.
In a generic model, the arrays in every field have to agree:

`````{tab-set}
````{tab-item} Python >= 3.12
```python
class Model[N: int, K: int](BaseModel):
    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, K], np.float64]

Model(vector=np.zeros(3), matrix=np.zeros((3, 4)))  # valid
Model(vector=np.zeros(3), matrix=np.zeros((4, 4)))  # invalid
```
````
````{tab-item} Python < 3.12
```python
N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)

class Model(BaseModel, Generic[N, K]):
    vector: NDArray[Shape[N], np.float64]
    matrix: NDArray[Shape[N, K], np.float64]

Model(vector=np.zeros(3), matrix=np.zeros((3, 4)))  # valid
Model(vector=np.zeros(3), matrix=np.zeros((4, 4)))  # invalid
```
````
`````

Parametrizing the model replaces its shape typevars,
so `Model[L[3], int]` has a `vector` of length 3 and a `matrix` with 3 rows.

This includes arrays in containers, like the items of a `list[NDArray[Shape[N]]]`.
Nested models have their own sizes.

```{note}
Shape typevars aren't represented in the generated JSON Schema.
JSON Schema can't express a size that depends on another dimension or another array
(see: https://github.com/orgs/json-schema-org/discussions/730).
```

```{important}
Type checkers don't treat a `TypeVar` as one size everywhere:
they accept arrays whose sizes disagree by widening the variable to cover all of them.
The [mypy plugin](mypy-plugin) refuses those calls.
See [](#shape-typevars-checkers) for what each type checker does,
and [this thread](https://discuss.python.org/t/support-function-with-multiple-generic-parameters-same-type/57931/15)
for details.
```

#### Constraints

Constraints behave like an "exact union" where a shape has to match exactly one value.
Each constraint has to be a size: `int`, a `Literal` int, a multi-value `Literal`,
a {class}`.Range`, or a union of them.

`````{tab-set}
````{tab-item} Python >= 3.12
```python
# these are equivalent: 
# (L[1], L[2, 3, 4, 5])
# (L[1], L[2] | L[3] | L[4] | L[5]), 
# (L[1], L[2], L[3], L[4], L[5])

def blend[
    RGB: (L[3], L[4]),  # 3 or 4
    N: (L[1], L[2, 3, 4, 5]),  # any one of 1 to 5
    Window: (Range[L[1], L[2]], Range[L[5], L[6]]),  # 1 to 2, or 5 to 6
](
    a: NDArray[Shape[N, Window, RGB], np.uint8],
    b: NDArray[Shape[N, Window, RGB], np.uint8],
) -> NDArray[Shape[N, Window, RGB], np.uint8]: ...

# valid
blend(np.zeros((2, 5, 3), dtype=np.uint8), np.zeros((2, 5, 3), dtype=np.uint8))  
# invalid: N is 2 and 3
blend(np.zeros((2, 5, 3), dtype=np.uint8), np.zeros((3, 5, 3), dtype=np.uint8))
# invalid: N as 3 is out of both ranges  
blend(np.zeros((3, 5, 3), dtype=np.uint8), np.zeros((3, 5, 3), dtype=np.uint8))
```
````
````{tab-item} Python < 3.12
```python
# these are equivalent:
# (L[1], L[2, 3, 4, 5]), 
# (L[1], L[2] | L[3] | L[4] | L[5]), 
# (L[1], L[2], L[3], L[4], L[5])

RGB = TypeVar("RGB", L[3], L[4])  # 3 or 4
N = TypeVar("N", L[1], L[2, 3, 4, 5])  # any one of 1 to 5
Window = TypeVar("Window", Range[L[1], L[2]], Range[L[5], L[6]])  # 1 to 2, or 5 to 6

def blend(
    a: NDArray[Shape[N, Window, RGB], np.uint8],
    b: NDArray[Shape[N, Window, RGB], np.uint8],
) -> NDArray[Shape[N, Window, RGB], np.uint8]: ...

# valid
blend(np.zeros((2, 5, 3), dtype=np.uint8), np.zeros((2, 5, 3), dtype=np.uint8))  
# invalid: N is 2 and 3
blend(np.zeros((2, 5, 3), dtype=np.uint8), np.zeros((3, 5, 3), dtype=np.uint8))
# invalid: N as 3 is out of both ranges  
blend(np.zeros((3, 5, 3), dtype=np.uint8), np.zeros((3, 5, 3), dtype=np.uint8))
```
````
`````

#### Bounds

A bound is the set of sizes the typevar can have.
Besides `int`, it can be a `Literal` int, a multi-value `Literal`, a {class}`.Range`,
or a union of them.

`````{tab-set}
````{tab-item} Python >= 3.12
```python
# a square matrix of at most 4 x 4
def invert[Small: Range[L[1], L[4]]](
    matrix: NDArray[Shape[Small, Small], np.float64],
) -> NDArray[Shape[Small, Small], np.float64]: ...

# 1, 3, or 4 channels, the same in every array that uses Channels
def blend[Channels: L[1, 3, 4]](
    a: NDArray[Shape[int, int, Channels], np.uint8],
    b: NDArray[Shape[int, int, Channels], np.uint8],
) -> NDArray[Shape[int, int, Channels], np.uint8]: ...
```
````
````{tab-item} Python < 3.12
```python
Small = TypeVar("Small", bound=Range[L[1], L[4]])
Channels = TypeVar("Channels", bound=L[1, 3, 4])

# a square matrix of at most 4 x 4
def invert(
    matrix: NDArray[Shape[Small, Small], np.float64],
) -> NDArray[Shape[Small, Small], np.float64]: ...

# 1, 3, or 4 channels, the same in every array that uses Channels
def blend(
    a: NDArray[Shape[int, int, Channels], np.uint8],
    b: NDArray[Shape[int, int, Channels], np.uint8],
) -> NDArray[Shape[int, int, Channels], np.uint8]: ...
```
````
`````

At runtime, a bound and constraints behave the same, since a shape typevar is one size everywhere.
Type checkers treat them differently: see [](#shape-typevars-checkers).

#### Bounds or constraints?

With runtime validation and the [mypy plugin](mypy-plugin), a bound and constraints
that allow the same sizes are checked the same way.
They only differ for other type checkers, so choose by what you want those to catch:

- For any size, use `bound=int`.
- For one of a few exact sizes, use constraints of single `Literal`s, like `L[3], L[4]`.
  Every type checker correctly refuses a size outside them cross-instance mismatches.
  A `Literal` bound, like `bound=L[3, 4]`, is refused when a size is outside it,
  but other type checkers pass cross-instance mismatches.
- For a range of sizes, use a {class}`.Range` bound.
  Other type checkers see a range as `int`, so only the mypy plugin and runtime validation
  check it. Use {class}`.Range` constraints only for ranges that don't overlap,
  like `Range[L[1], L[2]], Range[L[5], L[6]]`.
- In constraints, list sizes as separate `Literal`s, like `L[1], L[2], L[3]`,
  rather than one multi-value `Literal`, when other type checkers matter:
  they read `L[2, 3, 4, 5]` as a group of sizes that can differ.


### Composing shapes

Unpacked tuples are flattened into the shape,
so shapes can be built from aliases of other shapes:

```python
RGB = tuple[L[3]]
Frame = tuple[int, int]

Shape[*Frame, *RGB]  # Shape[int, int, L[3]]
```

Labelled dimensions are usually reused, so they can be given names:

```python
T = A[int, "t"]
X = A[L[1920], "x"]
Y = A[L[1080], "y"]
RGB = A[L[3], "rgb"]

Video = Shape[T, X, Y, RGB]
```

and composed:

```python
Frame = tuple[X, Y]

Video = Shape[T, *Frame, RGB]
```

### Bare integers

Integers can be used instead of `Literal` integers, like `Shape[3, 4]`.
This works at runtime and with the [mypy plugin](mypy-plugin),
but integers aren't valid in a type expression, so other type checkers will reject them.

## Annotated type with `NDArraySchema`

```{tip}
See also: [Typechecker Integration](typecheckers)
```

If you don't need compatibility with multiple array backends,
or want to have an array statically type check as a single array backend type,
Use the annotated schema form with {func}`.NDArraySchema`.

```python
from numpydantic import NDArraySchema
class MyModel(BaseModel):
    field: Annotated[np.ndarray, NDArraySchema(Shape[L[3], int], dtype)]
```

The shape can also be given as a tuple of its arguments, like `NDArraySchema((3, 4), dtype)`.

{func}`.NDArraySchema` also validates that the given array is of the specified type,
rather than any array backend that matches the dtype and shape.

## Deprecated string form

The string form of shapes, inherited from `nptyping`, is deprecated in v2.0
and will be removed in v3.0. It still works, but emits a {class}`DeprecationWarning`.
Strings aren't valid in a type expression, so only the [mypy plugin](mypy-plugin)
can type check them.

Each part of a string shape has an equivalent in the tuple form:

| String form            | Tuple form                                   |
|------------------------|----------------------------------------------|
| `Shape["3, 4"]`        | `Shape[L[3], L[4]]`                          |
| `Shape["3, *"]`        | `Shape[L[3], int]`                           |
| `Shape["2-4, 2-*, *-4"]` | `Shape[Range[L[2], L[4]], Range[L[2], int], Range[int, L[4]]]` |
| `Shape["3 x, * y"]`    | `Shape[A[L[3], "x"], A[int, "y"]]`           |
| `Shape["3, 4, ..."]`   | `Shape[L[3], L[4], *Repeat]`                |
| `Shape["*, ..."]`      | `Any`                                        |

The string can also be wrapped in a `Literal`, like `Literal["3, 4"]`,
split into one string per dimension, like `Shape["3", "*"]`,
or passed to the call form `Shape("3, 4")`.

```{warning}
The meaning of `...` is different in the string form:
in a string, `...` is any number of dimensions of any size,
but outside of a string, `...` repeats the dimension before it, as it does in `tuple[X, ...]`.

Use `*Repeat` for any number of dimensions of any size.
```

## Caveats

```{todo}
numpydantic currently does not support structured dtypes or {class}`numpy.recarray`
specifications like nptyping does. It will in future versions.
```
