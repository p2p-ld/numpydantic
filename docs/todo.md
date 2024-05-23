# TODO


## v2

We will be moving away from using nptyping in v2.0.0.

It was written for an older era in python before the dramatic changes in the Python
type system and is no longer actively maintained. We will be reimplementing a syntax
that extends its array specification syntax to include things like ranges and extensible
dtypes with varying precision (and is much less finnicky to deal with).


## Validation

```{todo}
Support pydantic value/range constraints - less than, greater than, etc.
```

```{todo}
Support different precision modes - eg. exact precision, or minimum precision
where specifying a float32 would also accept a float64, etc.
```

## Metadata

```{todo}
Use names in nptyping annotations in generated JSON schema metadata
```

## All TODOs

```{todolist}

``` 