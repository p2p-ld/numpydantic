---
file_format: mystnb
mystnb:
    output_stderr: remove
    render_text_lexer: python
    render_markdown_format: myst
    execution_allow_errors: true
    execution_show_tb: false
    code_prompt_show: "Show Exception"
myst:
    enable_extensions: ["colon_fence"]
---

# Dtype

```{todo}
This section is under construction as of 1.6.1

Much of the details of dtypes are covered in [syntax](./syntax.md)
and in {mod}`numpydantic.dtype` , but this section will specifically
address how dtypes are handled both generically and by interfaces
as we expand custom dtype handling <3.

For details of support and implementation until the docs have time for some love,
please see the tests, which are the source of truth for the functionality
of the library for now and forever.
```

Recall the general syntax:

```
NDArray[Shape, dtype]
```

These are the docs for what can do in `dtype`.

## Coercion

Pydantic is designed so that type annotations reflect the type that one can expect a field to be
*once the object has been instantiated and validated,* not the type that can be passed as input.

Accordingly, [pydantic attempts to coerce values](https://docs.pydantic.dev/latest/concepts/strict_mode/)
to their annotated types during validation when it can, 
rather than raising a validation error.

Numpydantic follows this general pattern when it is feasible,
but attempting to coerce the shape or dtype of an array is *significantly* more costly, 
and much more likely to result in incorrect data than the scalar data pydantic was designed for.

It tries to balance *usability* with *minimization of surprise* -
a primary design goal is to be a transparent passthrough validator system 
that gets out of the way of the underlying array libraries
so they can be used as expected.

Numpydantic **will**

- Attempt to coerce non-array values like scalars or python sequences to numpy arrays when possible.
  Validation against shape and dtype happen after coercion.
- Attempt to coerce any recognized {meth}`~.Interface.input_types` to the {meth}`~.Interface.return_type`
  of the matching array interface (see [Interface Matching](./interfaces.md#matching))
  (E.g. the {class}`.VideoInterface` recognizes {class}`~pathlib.Path` and string objects with supported file extensions
  and coerces them to a {class}`.VideoProxy` arraylike object).
- Attempt to coerce lists or arrays of `dict`s to the annotated {class}`~pydantic.BaseModel`,
  when the annotation is a pydantic model (see [model dtypes](#model-dtypes))

Numpydantic **will not**

- Attempt to reshape an input array
- Attempt to change the dtype of an array

## Scalar Dtypes

Python builtin types and numpy types should be handled transparently,
with some exception for complex numbers and objects (described below).

The {mod}`numpydantic.testing` subpackage, and more specifically the
{mod}`numpydantic.testing.cases` module are the source of truth for currently supported
types and their behavior. Each case is tested combinatorically against each of the `shape`
cases and array interfaces.

### Numbers

Specific dtypes can be specified as single dtype classes.

```{code-cell}
from typing import Any
import numpy as np
from numpydantic import NDArray
from pprint import pprint

UInt8Array = NDArray[Any, np.uint8]
UInt8Array(np.array([1,2,3], dtype=np.uint8))
```

```{code-cell}
:tags: [hide-output]

UInt8Array(np.array([1,2,3], dtype=np.uint32))
```

numpydantic interprets the builtin `float` and `int` a bit differently than `numpy` - 
`numpy` treats `float` as equivalent to {class}`numpy.float64` 
and `int` as {class}`numpy.int64` when used in array creation.

```{code-cell}
print(
    np.array([1,2,3], dtype=float).dtype,
    np.array([1,2,3], dtype=int).dtype
)
```

Numpydantic treats `float` and `int` as *any float* or *any integer*[^icanchange],
since it is the parsimonious way to express "any float/integer" 
when thinking across, rather than within a single array library,
and is common need in data standards.
`numpy.int64` and `float64` already have specific dtypes! they are them!

```{code-cell}
print(
    NDArray[Any, int]([1,2,3]),
    NDArray[Any, int](np.array([1,2,3], dtype=np.uint8)),
    NDArray[Any, int](np.array([1,2,3], dtype=np.int16)),
)
```

Numpydantic provides a handful of aliases to numpy dtypes so they appear more
"Annotation-like", and a handful of "generic" types like {data}`~numpydantic.dtype.Float`
and {data}`~numpydantic.dtype.UnsignedInteger`
(which are just tuples of numpy dtypes),
but in general it is recommended to just use numpy dtypes and type unions directly.

Many array libraries that are not numpy understand numpy dtypes.
If you are using an array library that doesn't,
you can use [type unions](#union-types) to express whatever combination of types you'd like.

#### Complex numbers

```{todo}
Document limitations for complex numbers and strategies for serialization/validation
```

### Datetimes

```{todo}
Datetimes are supported by every interface except :class:`.VideoInterface` ,
with the caveat that HDF5 loses timezone information, and thus all timestamps should
be re-encoded to UTC before saving/loading. 

More generic datetime support is TODO.
```

### Objects

Generic objects are supported by all interfaces except
{class}`.VideoInterface` , {class}`.HDF5Interface` , and {class}`.ZarrInterface` .

this might be expected, but there is also hope.

When the numpy interface validates arrays of objects, 
it only checks the first item in the array for object identity
(`type(array.flat()[0]) is dtype`),
as iterating through every object in an array would be downright silly levels of expensive for default behavior.

PRs welcome for implementing opt-in strict object validation behavior.

### Strings

```{todo}
Strings are supported by all interfaces except :class:`.VideoInterface` .

TODO is fill in the subtleties of how this works
```

## Model Dtypes

Pydantic models can be used as dtypes, and numpydantic will attempt to cast them to the model class
when passed as a `dict`, e.g. when validating from JSON.

```{code-cell}
from pydantic import BaseModel

class KimPetras(BaseModel):
    listen: str = "to"
    turn: str = "off"
    the: str = "light"
    n: int = 5000
    times: str = "!"

NDArray[Any, KimPetras]([
    {"listen": "up", "turn": "s out", "the": "album is good"},
    {"n": "10000", "times": "is more like it"},
])
```

Models are supported in the same set of array interfaces that 


### Union Types

Union types can be used as expected.

```{code-cell}
from typing import Union

print(
  NDArray[Any, np.float16 | np.int32](np.array([1,2,3], dtype=np.float16)),
  NDArray[Any, Union[np.float16, np.int32]](np.array([1,2,3], dtype=np.int32))
)
```

Since union types are cumbersome to work with, unions can also be expresses as tuples of types

```{code-cell}
NDArray[Any, (np.uint8, np.uint16)](np.array([1,2,3], dtype=np.uint8))
```

Python unnests nested union types automatically,
and numpydantic tests against tuple unions recursively --
any item at any level can match.

```{code-cell}
from numpydantic.dtype import SignedInteger
CoolTypes = (np.uint8, np.datetime64)
AlsoCool = (np.str_, SignedInteger)

Faves = (CoolTypes, AlsoCool)
pprint(Faves)
```

```{code-cell}
NDArray[Any, Faves](np.array([1,2,3], dtype=np.int16))
```


## Arbitrary Dtypes

Numpydantic does *not* support every possible type that python is able to express as a dtype,
mostly because array libraries can't support every possible type. 
Arrays are usually numbers and strings, and more complex dtypes can be expressed as pydantic models.

Validating elaborated objects like an array of `tuple[int, str]`,
or pydantic annotated types with validation functions would involve
iterating through every element of an array, 
except for a small subset of cases like `annotated_types.Gt()`
and other annotations that have common vectorized operations.

PRs are welcome for implementing annotated types support,
or for any other elaborated type that isn't currently supported but is a common or useful dtype :).

## Compound Dtypes

```{todo}
Compound dtypes are currently unsupported,
though the HDF5 interface supports indexing into compound dtypes
as separable dimensions/arrays using the third "field" parameter in
{class}`.hdf5.H5ArrayPath` .
```

[^icanchange]: on this, and all design decisions,
  i am happy to hear criticism and entertain changes to behavior if this is too surprising.
  feel free to raise an issue!
