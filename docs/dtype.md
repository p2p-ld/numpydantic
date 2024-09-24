# dtype

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

## Scalar Dtypes

Python builtin types and numpy types should be handled transparently,
with some exception for complex numbers and objects (described below).

### Numbers

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

```{todo}
Generic objects are supported by all interfaces except
:class:`.VideoInterface` , :class;`.HDF5Interface` , and :class:`.ZarrInterface` .

this might be expected, but there is also hope, TODO fill in serialization plans.
```

### Strings

```{todo}
Strings are supported by all interfaces except :class:`.VideoInterface` .

TODO is fill in the subtleties of how this works
```

## Generic Dtypes

```{todo}
For now these are handled as tuples of dtypes, see the source of
{ref}`numpydantic.dtype.Float` . They should either be handled as Unions
or as a more prescribed meta-type.

For now, use `int` and `float` to refer to the general concepts of
"any int" or "any float" even if this is a bit mismatched from the numpy usage.
```

## Extended Python Typing Universe

### Union Types

Union types can be used as expected. 

Union types are tested recursively -- if any item within a ``Union`` matches
the expected dtype at a given level of recursion, the dtype test passes.

```python
class MyModel(BaseModel):
    array: NDArray[Any, int | float]
```

## Compound Dtypes

```{todo}
Compound dtypes are currently unsupported,
though the HDF5 interface supports indexing into compound dtypes
as separable dimensions/arrays using the third "field" parameter in
{class}`.hdf5.H5ArrayPath` .
```


