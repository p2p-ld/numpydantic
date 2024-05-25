# TODO


## v2

We will be moving away from using nptyping in v2.0.0.

It was written for an older era in python before the dramatic changes in the Python
type system and is no longer actively maintained. We will be reimplementing a syntax
that extends its array specification syntax to include things like ranges and extensible
dtypes with varying precision (and is much less finnicky to deal with).

(type_checkers)=
## Type Checker Integration

The `.pyi` stubfile generation ({mod}`numpydantic.meta`) works for 
keeping type checkers from complaining about various array formats
not literally being `NDArray` objects, but it doesn't do the kind of 
validation we would want to be able to use `NDArray` objects as full-fledged
python types, including validation propagation through scopes and 
IDE type checking for invalid literals.

We want to hook into the type checking process to satisfy these type checkers:
- mypy - has hooks, can be done with an extension
- pyright - unclear if has hooks, might nee to monkeypatch
- pycharm - unlikely this is possible, extensions need to be in Java and installed separately


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