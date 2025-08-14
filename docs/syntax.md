# Syntax

General form:

```python
field: NDArray[Shape["{shape_expression}"], dtype]
```

For better compatibility with static type checkers,
rather than `Shape` with a string literal, you can use {class}`typing.Literal`
anywhere you can use `Shape`.

```python
field: NDArray[Literal["{shape_expression"], dtype]
```

## Dtype

Dtype checking is for the most part as simple as an `isinstance` check - 
the `dtype` attribute of the array is checked against the `dtype` provided in the
`NDArray` annotation. Both numpy and builtin python types can be used.

A tuple of types can also be passed:

```python
field: NDArray[Shape["2, 3"], (np.int8, np.uint8)]
```

Like `nptyping`, the {mod}`~numpydantic.dtype` module provides convenient access
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

Full documentation of nptyping's shape syntax is available in the [nptyping docs](https://github.com/ramonhagenaars/nptyping/blob/master/USERDOCS.md#Shape-expressions),
but for the sake of self-contained docs, the high points are:

### Numerical Shape

A comma-separated list of integers. 

For a 2-dimensional, 3 x 4-shaped array:

```python
Shape["3, 4"]
```

### Wildcards

Wildcards indicate a dimension can be any size

For a 2-dimensional, 3 x any-shaped array:

```python
Shape["3, *"]
```

(shape-ranges)=
### Ranges

Dimension sizes can also be specified as ranges[^ranges].
Ranges must have no whitespace, and may use integers or wildcards.
Range specifications are **inclusive** on both ends.

For an array whose...
- First dimension can be of length 2, 3, or 4
- Second dimension is 2 or greater
- Third dimension is 4 or less

```python
Shape["2-4, 2-*, *-4"]
```

[^ranges]: This is an extension to nptyping's syntax, and so using `nptyping.Shape` is unsupported - use {class}`numpydantic.Shape`

### Labels

Dimensions can be given labels, and in future versions these labels will be 
propagated to the generated JSON Schema

```python
Shape["3 x, 4 y, 5 z"]
```

### Arbitrary dimensions

After some specified dimensions, one can express that there can be any number
of additional dimensions with an `...` like

```python
Shape["3, 4, ..."]
```

### Any-Shaped

If `dtype` is also `Any`, one can just use 

```python
field: NDArray
```

If a `dtype` is being passed, use the `'*'` wildcard along with the `'...'` 

```python
field: NDArray[Shape['*, ...'], int]
```

## Caveats

```{todo}
numpydantic currently does not support structured dtypes or {class}`numpy.recarray`
specifications like nptyping does. It will in future versions.
```

````{todo}
numpydantic also does not support the variable shape definition form like

```python
Shape['Dim, Dim']
```

where there are two dimensions of any shape as long as they are equal
because at the moment it appears impossible to express dynamic constraints
(ie. `minItems`/`maxItems` that depend on the shape of another array)
in JSON Schema. A future minor version will allow them by generating a JSON
schema with a warning that the equal shape constraint will not be represented.

See: https://github.com/orgs/json-schema-org/discussions/730

````