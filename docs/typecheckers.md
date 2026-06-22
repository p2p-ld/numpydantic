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

Enable the mypy plugin in your pyproject.toml configuration

```toml
[tool.mypy]
plugins = [
  "numpydantic.mypy",
]
```

And configure it with with the `tool.numpydantic.mypy` table

```toml
[tool.numpydantic.mypy]
interfaces = [
  "numpy",
  "dask",
  "zarr",
]
```

If you are using numpydantic with pydantic, you should also enable [pydantic's mypy plugin](https://pydantic.dev/docs/validation/latest/integrations/dev-tools/mypy/).
By default pydantic uses `Any` types for all the fields in its synthesized `__init__` method,
so static checking for array types doesn't work.
Numpydantic's plugin handles its input type transformations correctly (see [below](#pydantic-models)), so you likely want to set its `init_typed` option to `True`.

If you are using numpydantic with zarr or any other array interface whose package is untyped,
you will need to enable mypy's [`follow-untyped-imports`](https://mypy.readthedocs.io/en/stable/running_mypy.html) option.

A full configuration might then look like this:

```toml
[tool.mypy]
plugins = [
  "numpydantic.mypy",
  "pydantic.mypy",
]

[[tool.mypy.overrides]]
module = ["zarr.*"]
follow_untyped_imports = true

[tool.numpydantic.mypy]
interfaces = [
  "numpy",
  "dask",
  "zarr",
]

[tool.pydantic-mypy]
init_typed = true
```

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
:lines: 6-
:linenos:
:lineno-start: 6
```

```{program-output} mypy --pretty examples/correct/shape_scalar.py
```

**Incorrect**

```{literalinclude} examples/incorrect/shape_scalar.py
:lines: 6-
:linenos:
:lineno-start: 6
```

```{program-output} mypy --pretty --color-output examples/incorrect/shape_scalar.py
:returncode: 1
```

#### Ranges

Ranges work on the left hand side with a right hand side scalar (see [known limitations](#range-range-checking)).
When possible, you should attempt to keep right-hand/return annotations scalar for the strongest typing,
even if left-hand annotations can accept shape ranges.

**Correct**

```{literalinclude} examples/correct/shape_range_scalar.py
:lines: 6-
:linenos:
:lineno-start: 6
```

```{program-output} mypy --pretty examples/correct/shape_range_scalar.py
```

**Incorrect**
 
```{literalinclude} examples/incorrect/shape_range_scalar.py
:lines: 6-
:linenos:
:lineno-start: 6
```

```{program-output} mypy --pretty --color-output examples/incorrect/shape_range_scalar.py
:returncode: 1
```  

#### Wildcards & Ellipses

As during runtime, wildcards specify that an array dimension must exist but can be any size,
and ellipses specify that any number of additional dimensions may be included.

**Correct**

```{literalinclude} examples/correct/shape_wildcard.py
:lines: 7-
:linenos:
:lineno-start: 7
```

```{program-output} mypy --pretty examples/correct/shape_wildcard.py
```

**Incorrect**

```{literalinclude} examples/incorrect/shape_wildcard.py
:lines: 7-
:linenos:
:lineno-start: 7
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
from typing import Annotated as A
import zarr

from numpydantic import NDArraySchema, Shape

def make_zarr() -> A[zarr.Array, NDArraySchema(Shape(1, 2, 3))]:
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
:lines: 7-
:linenos:
:lineno-start: 7
```

```{program-output} mypy --pretty examples/correct/pydantic_field.py
```

**Incorrect**

```{literalinclude} examples/incorrect/pydantic_field.py
:lines: 7-
:linenos:
:lineno-start: 7
```

```{program-output} mypy --pretty examples/incorrect/pydantic_field.py
:returncode: 1
```



### Known Limitations

#### Range-range checking

Our implementation of ranges is [somewhat cursed](https://github.com/python/mypy/issues/16497#issuecomment-4570099557),
and it is currently impossible to check if a right-hand side range is contained within a left-hand range.
This is because we check ranges against literals using `__eq__` which is commutative.

E.g. the following should fail, but it does not:

```{literalinclude} examples/correct/shape_range_range.py
```

```{program-output} mypy --pretty --color-output examples/correct/shape_range_range.py
```  

Pull requests welcome!

## pyright

Pyright can only use the standard types and the type stubs by design.

You will have to provide all the type annotations yourself,
and constructs that can't be expressed in the python type system liks ranges can't be used.

In general, you have to specify types using tuples and literals, like:

```python
def some_function(array: NDArray[tuple[Literal[3], Literal[3]], np.uint8]) -> None: ...
```