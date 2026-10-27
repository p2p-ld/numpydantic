# Typechecker Integration

Numpydantic does things with the python type system that are not formally supported,
it tries to have permissive behavior by default for typecheckers,
but provides plugins for strict static type checking.

By default, for non-mypy typecheckers, 
the {class}`.NDArray` class will appear to be a {class}`numpy.ndarray` . 
To support non-mypy arrays, you will have to use the {func}`.NDArraySchema` function
as an annotated type, like `Annotated[zarr.Array, NDArraySchema()]`

(mypy-plugin)=
## Mypy Plugin

```{versionadded} 1.9.0
```

```{admonition} Experimental!
:class: warning

The mypy plugin is experimental!
Please raise issues describing any bugs or shortcomings!
Input is very welcome on what would be useful here.
```

Numpydantic provides a mypy plugin that is capable of:

- Checking {class}`.NDArray`-{class}`.NDArray` annotation compatibility with shapes and dtypes
- Checking {class}`numpy.ndarray`-{class}`.NDArray` annotations: blank `np.ndarray` annotations
  are treated as `NDArray[Any, Any]`
- Inferring array shapes and dtypes from constructors like {func}`numpy.ones` for any supported interfaces when arrays are constructed with integer literals.

Type checking works across all typing comparisons: 
directly annotated assignments,
function/method parameters and return types,
and of course pydantic fields! (see [below](#pydantic-models))

This makes it possible to escape some of the ambiguity of passing `np.ndarray` types around
with explicit annotations!

E.g. say you have some analysis function that only accepts grayscale images:

```{literalinclude} examples/incorrect/rgb_gray_frame.py
:linenos:
```

Find bugs before they're deployed with your type checker!

```{program-output} mypy --pretty examples/incorrect/rgb_gray_frame.py
:returncode: 1
```

### Configuration

Enable the mypy plugin in your `pyproject.toml` or `mypy.ini` configuration

`````{tab-set}
````{tab-item} pyproject.toml
```toml
[tool.mypy]
plugins = [
  "numpydantic.mypy",
]
```
````
````{tab-item} mypy.ini
```ini
[mypy]
plugins=numpydantic.mypy
```
````
`````

And configure it with with the `tool.numpydantic.mypy` table in `pyproject.toml`, or `numpydantic-mypy` in `mypy.ini`

`````{tab-set}
````{tab-item} pyproject.toml
```toml
[tool.numpydantic.mypy]
interfaces = [
  "numpy",
  "dask",
  "zarr",
]
```
````
````{tab-item} mypy.ini
```ini
[numpydantic-mypy]
interfaces=numpy,dask,zarr
```
We follow mypy's convention of using comma-separated values rather than newlines
````
`````

If you are using numpydantic with pydantic, you should also enable [pydantic's mypy plugin](https://pydantic.dev/docs/validation/latest/integrations/dev-tools/mypy/).
By default pydantic uses `Any` types for all the fields in its synthesized `__init__` method,
so static checking for array types doesn't work.
Numpydantic's plugin handles its input type transformations correctly (see [below](#pydantic-models)), so you likely want to set its `init_typed` option to `True`.

If you are using numpydantic with an array interface whose package is untyped,
you will need to enable mypy's [`follow-untyped-imports`](https://mypy.readthedocs.io/en/stable/running_mypy.html) option for that package.

A full configuration might then look like this:

`````{tab-set}
````{tab-item} pyproject.toml
```toml
[tool.mypy]
plugins = [
  "numpydantic.mypy",
  "pydantic.mypy",
]

[tool.numpydantic.mypy]
interfaces = [
  "numpy",
  "dask",
  "zarr",
]

[tool.pydantic-mypy]
init_typed = true
```
````
````{tab-item} mypy.ini
```ini
[mypy]
plugins=numpydantic.mypy,pydantic.mypy

[numpydantic-mypy]
interfaces=numpy,dask,zarr

[pydantic-mypy]
init_typed=true
```
````
`````

```{important}
The `numpydantic.mypy` plugin **must** come before the `pydantic.mypy` plugin in the list - 
mypy only allows one plugin to respond to a hook call,
so we extend and replace how pydantic decorates `__init__` methods.
```

#### Configuration Reference

```{eval-rst}
.. autoclass:: numpydantic.mypy.plugin_.MypyPluginOptions
  :members:
  :exclude-members: from_options
```


### Shape Checking

#### Scalars

Scalar shapes are checked as expected, where the shapes must match exactly.

**Correct**

```{literalinclude} examples/correct/shape_scalar.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty examples/correct/shape_scalar.py
```

**Incorrect**

```{literalinclude} examples/incorrect/shape_scalar.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty --color-output examples/incorrect/shape_scalar.py
:returncode: 1
```

#### Ranges

An exact size can be assigned to a {class}`.Range` that contains it (see [known limitations](#range-checking)).
When possible, you should attempt to keep right-hand/return annotations scalar for the strongest typing,
even if left-hand annotations can accept shape ranges.

**Correct**

```{literalinclude} examples/correct/shape_range_scalar.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty examples/correct/shape_range_scalar.py
```

**Incorrect**
 
```{literalinclude} examples/incorrect/shape_range_scalar.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty --color-output examples/incorrect/shape_range_scalar.py
:returncode: 1
```  

#### Any size & any number of dimensions

As during runtime, `int` specifies that an array dimension must exist but can be any size,
and `*AnyDims` specifies that any number of additional dimensions may be included.

**Correct**

```{literalinclude} examples/correct/shape_wildcard.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty examples/correct/shape_wildcard.py
```

**Incorrect**

```{literalinclude} examples/incorrect/shape_wildcard.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --pretty examples/incorrect/shape_wildcard.py
:returncode: 1
```

### Dtype Checking

**Correct**

```{literalinclude} examples/correct/dtype_basic.py
:lines: 5-
:linenos:
:lineno-start: 5
```

```{program-output} mypy --pretty examples/correct/dtype_basic.py
```

**Incorrect**

```{literalinclude} examples/incorrect/dtype_basic.py
:lines: 5-
:linenos:
:lineno-start: 5
```

```{program-output} mypy --pretty examples/incorrect/dtype_basic.py
:returncode: 1
```


### Constructor Inference

Constructor inference works by modifying the type returned from supported array constructors:

```{literalinclude} examples/correct/numpy_inference.py
:lines: 5-

```

**Without the plugin**

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/correct/numpy_inference.py
```

**With the plugin**

```{program-output} mypy --pretty examples/correct/numpy_inference.py
```

Each interface may support constructor inference by declaring a {class}`~.interface.typing.InterfaceTyping` class with a set of {class}`~.interface.typing.ConstructorSpec` objects.
You can see the currently supported constructors on the relevant interface pages
(e.g. for [numpy](numpydantic.interface.numpy.NumpyTyping), [zarr](numpydantic.interface.zarr.ZarrTyping)).

The {class}`~.interface.typing.ConstructorSpec`s declare how to locate the shape and dtype args or kwargs,
(these are almost always shape in the first positional arg and the dtype specified as a kwarg,
but why assume!)

So, if enabled, return values for non-numpy interfaces can declare how to infer their shapes and dtypes:

```{literalinclude} examples/correct/interface_inference.py
```

**Without the plugin**

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/correct/interface_inference.py
```

**With the plugin**

```{program-output} mypy --pretty examples/correct/interface_inference.py
```

```{admonition} Dask is broken
:class: tip

Note that dask's constructor inference doesn't work at the moment.
This is due to dask's array creation routines being positively haunted,
an untyped wrapped dynamic construction of a function that creates a class
that creates a class.

PRs welcome re: figuring out how to type that.
```
 
````{todo}
This is less than optimal. Even though the inferred types from the constructors are `Any`
without the plugin, and so being typed with numpy methods is better than nothing,
we are missing the backend-specific types.

In the future we will be extending the mypy plugin to understand the {func}`.NDArraySchema`-style
annotated types, PRs welcome!

```python
from typing import Annotated as A, Literal as L
import zarr

from numpydantic import NDArraySchema, Shape

def make_zarr() -> A[zarr.Array, NDArraySchema(Shape[L[1], L[2], L[3]])]:
    return zarr.zeros((1,2,3))
```
````


### Pydantic Models

When used as a type on a pydantic model,
numpydantic is able to coerce convenience input types into arrays.
This means that we should consider some decidedly non-array inputs as satisfying
an array type - like paths and strings - which makes sense for pydantic models,
but would be bad to accept as a function param.

The mypy plugin can detect when the annotation is being used within a pydantic model,
and allows the items within the enabled interfaces {meth}`~.Interface.input_types` list,
and otherwise refuses them.

When checking non-array input types like a path, shape and dtype checking is unsupported,
as it would be an absolutely absurd thing to do to open on-disk array stores or analyze videos while type checking.

**Correct**

```{literalinclude} examples/correct/pydantic_field.py
:lines: 9-
:linenos:
:lineno-start: 9
```

```{program-output} mypy --pretty examples/correct/pydantic_field.py
```

**Incorrect**

```{literalinclude} examples/incorrect/pydantic_field.py
:lines: 9-
:linenos:
:lineno-start: 9
```

```{program-output} mypy --pretty examples/incorrect/pydantic_field.py
:returncode: 1
```



### Known Limitations

#### Range checking

Our implementation of ranges is [somewhat cursed](https://github.com/python/mypy/issues/16497#issuecomment-4570099557):
mypy compares literals using `__eq__`, so that is where ranges are checked.

A range must fit within the range it is assigned to:

```{literalinclude} examples/incorrect/shape_range_range.py
```

```{program-output} mypy --pretty --color-output examples/incorrect/shape_range_range.py
:returncode: 1
```

But when comparing a range with an exact size, we can't tell which side of the
assignment the range is on. An exact size can be assigned to a range that contains it,
but a range can also be assigned to an exact size it contains,
e.g. `Range[L[2], L[4]]` can be assigned to `L[3]`, even though it shouldn't be.

Pull requests welcome!

## pyright

Pyright can only use the standard types and the type stubs by design.

You will have to provide all the type annotations yourself.
To pyright, {class}`.Shape` is just `tuple` and a {class}`.Range` is just `int`,
so ranges can be used but aren't checked.
The deprecated string form and bare integers like `Shape[3, 3]` can't be used.

```python
def some_function(array: NDArray[Shape[Literal[3], Literal[3]], np.uint8]) -> None: ...
```

Additionally, you must always specify dtypes as numpy types rather than builtins like `float`.
This is strictly correct according to numpy's type annotations,
which only accept subtypes of `np.dtype`,
even if by convention builtins like `float` and `int` are accepted in constructors.

## mypy and numpy's overloads

See [](unshaping.md) for how to handle the shapes numpy gives.
mypy handles some of numpy's overloads differently from pyright:

### Constructors assigned directly

Without the plugin, mypy accepts any shape for a constructor assigned straight to an annotation,
because it falls back to a numpy overload that gives an unknown shape.
Pyright and the mypy plugin check it:

```{literalinclude} examples/numpy/mypy_context.py
```

**Without the plugin**

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/mypy_context.py
```

**With the plugin**

```{program-output} mypy --pretty examples/numpy/mypy_context.py
:returncode: 1
```

### Nested numpy calls

When a numpy function is called directly inside {func}`.unliteral` or {func}`.transposed`,
like `transposed(np.transpose(a))`, mypy gives an unknown shape, which is accepted anywhere.
mypy picks numpy's overload using the first of the helper's overloads, which is for unknown shapes.
Assign numpy's result to a variable first, or use `a.T`.
