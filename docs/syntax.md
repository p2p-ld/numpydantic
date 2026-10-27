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
from numpydantic import AnyDims, NDArray, Range, Shape
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

### Any number of dimensions

{data}`.AnyDims` is any number of dimensions of any size, including none.
Unpack it as the last item in a shape, after any other dimensions:

```python
# at least two dimensions, the first two of which are 3 x 4
Shape[L[3], L[4], *AnyDims]
```

`*` unpacking in a subscript needs python 3.11 or later -
before that, use `Unpack`:

```python
from typing_extensions import Unpack

Shape[L[3], L[4], Unpack[AnyDims]]
```

### Repeated dimensions

As with `tuple[X, ...]`, `Shape[X, ...]` is any number of dimensions matching `X`:

```python
# any number of dimensions of length 2, including none
Shape[L[2], ...]
```

After more than one dimension, `...` repeats the last one.
This is not valid for `tuple`, so only the mypy plugin understands it statically:

```python
# a 1-length dimension, a 2-length dimension, and any number of additional 2s
Shape[L[1], L[2], ...]
# the same, as a valid type expression
Shape[L[1], L[2], *tuple[L[2], ...]]
```

### Labels

Dimensions can be given labels with {class}`typing.Annotated`.
They don't change the shape, and in future versions they will be
propagated to the generated JSON Schema.

```python
Shape[A[L[3], "x"], A[L[4], "y"], A[int, "z"]]
```

### Any shape

If `dtype` is also `Any`, one can just use 

```python
field: NDArray
```

If a `dtype` is being passed, use `Any` or a bare `Shape` for the shape:

```python
field: NDArray[Any, int]
# or
field: NDArray[Shape, int]
```

`Shape[int, ...]` and `Shape[*AnyDims]` also mean any shape.

### Zero-dimensional arrays

As with `tuple[()]`, an empty shape is a zero-dimensional array:

```python
Shape[()]
```

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
| `Shape["3, 4, ..."]`   | `Shape[L[3], L[4], *AnyDims]`                |
| `Shape["*, ..."]`      | `Any`                                        |

The string can also be wrapped in a `Literal`, like `Literal["3, 4"]`,
split into one string per dimension, like `Shape["3", "*"]`,
or passed to the call form `Shape("3, 4")`.

```{warning}
The meaning of `...` is different in the string form:
in a string, `...` is any number of dimensions of any size,
but outside of a string, `...` repeats the dimension before it, as it does in `tuple[X, ...]`.

Use `*AnyDims` for any number of dimensions of any size.
```

## Caveats

```{todo}
numpydantic currently does not support structured dtypes or {class}`numpy.recarray`
specifications like nptyping does. It will in future versions.
```

````{todo}
numpydantic also does not support shape variables, where dimensions
must be the same size as each other, like nptyping's string form

```python
Shape['Dim, Dim']
```

These will be supported with {class}`typing.TypeVar` and {class}`typing.TypeVarTuple`
in a future version. JSON Schema can't express this kind of constraint
(ie. `minItems`/`maxItems` that depend on the shape of another array),
so it will not be represented in the generated JSON Schema.

See: https://github.com/orgs/json-schema-org/discussions/730

````
