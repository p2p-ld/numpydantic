# Unshaping & Casting

Most of the array world does not have strict, literal shape annotations!

When shape information *is* available from external methods,
it is often only as `tuple[int, ...]`s - just the cardinality, not the exact shape.
Since `int` is a superset of a `Literal` like `Literal[3]`, 
type checkers correctly refuse assigning a more general shape to a specific one:

```{literalinclude} examples/numpy/generic_assign.py
:lines: 7-
:linenos:
:lineno-start: 7
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/generic_assign.py
:returncode: 1
```

In most cases where array shape is mutated,
including in many of numpy's core array operations,
the resulting shape can't be correctly inferred by python's type system:

```{literalinclude} examples/numpy/reshape.py
:lines: 7-
:linenos:
:lineno-start: 7
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/reshape.py
:returncode: 1
```

The way we *want* to be able to use numpydantic types is as *type narrowing* annotations,
but the python type system doesn't really cooperate with that.

E.g. we *want* 

```python
x: NDArray[Shape[L[2], L[3]]] = np.zeros((2, 3))
```

to mean "Numpy doesn't know what the shape is, but I am telling you, the typechecker, that I know it is (2, 3)."
However to the typechecker, there is no difference between that statement and the exact case we want to check for,
where our strict annotation should fail when we call it with an array we aren't sure matches the annotation.

This requires us to be explicit with which part of the [Rumsfeldian quartet](https://en.wikipedia.org/wiki/There_are_unknown_unknowns)
we are on: 
- Known knowns: we know that we correctly know the literal shape and should check it accordingly
- Unknown knowns: [`typing.cast`](#cast) and [plain annotations](#annotation) - the type checker doesn't know the shape, but *we* know the shape, and give it an explicit annotation.
- Known unknowns: [`unshape`](#unshape) / [`unliteral`](#numpy-kept-sizes-that-changed)  - we explicitly affirm that we *don't* know the literal shape and should be permissive when accepting it and allow its type to be narrowed
- Unknown unknowns: we don't know the shape and don't know what to do with it, so the typechecker fails and alerts us that more information is needed

Numpydantic has a few helpers to tell the type checker what it can't work out.
They are no-ops at runtime, don't touch the array, and only modify the way its type is checked.

The general pattern of use is similar to the [robustness principle](https://en.wikipedia.org/wiki/Robustness_principle):
Be liberal in what you accept from external packages - 
assume they are not strictly shaped, recast them, and then be strict about shapes internally and for downstream consumers.

```{admonition} Numpy's shape typing is unstable!
:class: warning

As of the time of writing, numpy's shape typing is in flux and should be considered unstable.
The general *form* of the ndarray annotations is settling out, but the behavior of individual functions and methods has some work to do.
E.g. 2.5 introduces returning literal tuple types from some constructor, but then [2.6 is scheduled to remove them](https://github.com/numpy/numpy/issues/33150#issuecomment-6427676020)
```

(annotation)=
## Annotation - enrich unknown shapes

An array with an `Any` type or an `Any` shape, such as those from {func}`numpy.load` and {func}`numpy.loadtxt`,
can be annotated directly:

```{literalinclude} examples/numpy/unknown.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/unknown.py
```

(cast)=
## Cast - force the type

{func}`typing.cast` is the standard way to insist that you know better what a type is to a rambunctious type checker.

```{literalinclude} examples/numpy/cast.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/cast.py
```

This completely overwrites the type, but can mask real errors - 
a plain `cast` can hide when the input type is not an array at all,
and the author has to manually assure that the dtype matches on both sides.

(unshape)=
## Unshape - a safer cast

The {func}`~numpydantic.shape.unshape` helper is useful whenever numpy returns shapes as plain ints and you want to cast them to a more specific shape.

It resets the shape of the array and passes through its dtype, allowing its shape to be recast later.
`unshape` combined with an annotation is safer than a plain `cast` because 
a) the input arg is checked as being an `ndarray`, and
b) the dtypes are checked as matching.

```{literalinclude} examples/numpy/unshape.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/unshape.py
```

(unliteral)=
## Unliteral - handling unknown shape mutations

In circumstances where the shape of an array is mutated,
and it's possible to know the number of dimensions but not any of their sizes,
{func}`~numpydantic.shape.unliteral` preserves the dimensionality but removes literal shapes.

```{literalinclude} examples/numpy/unliteral.py
:lines: 8-
:linenos:
:lineno-start: 8
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/unliteral.py
```

## Transposed arrays

Most shape mutations require explicit annotation, but not all of them!
Some operations like {func}`numpy.transpose` or {meth}`~numpy.ndarray.T`
have predictable changes in shape.

Wrap {func}`~numpydantic.shape.transposed` around a transposition to propagate the shape correctly.


numpy types functions that reorder dimensions as returning the shape they were given,
which is only right when every dimension is the same size.
Wrap `.T`, `.transpose()` or `np.transpose(a)` with {func}`.transposed` to reverse the shape,
for up to 5 dimensions.


```{literalinclude} examples/numpy/transpose.py
```

```{program-output} mypy --config-file examples/blank_config.toml --pretty examples/numpy/transpose.py
```

We'll be adding more convenience methods like this as-needed.

```{warning}
With mypy, assign `np.transpose(a)` to a variable first (see [](typecheckers.md#nested-numpy-calls)).
For other reorderings, like {func}`numpy.swapaxes` or {func}`numpy.moveaxis`,
use `unshape` and annotate the shape.
```