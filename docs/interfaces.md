# Interfaces

Interfaces are the bridge between the abstract {class}`~numpydantic.NDArray` specification
and concrete array libraries. They are subclasses of the abstract {class}`.Interface`
class.

They contain methods for coercion, validation, serialization, and any other 
implementation-specific functionality. 

## Discovery

Interfaces are discovered through the {meth}`.Interface.interfaces` method - 
returning all subclasses of `Interface`. To use a custom interface, it just
needs to be defined/imported by the time you intend to use it when instantiating
a pydantic model.

Each interface implements a {meth}`.Interface.enabled` method that determines
whether that interface can be used. Typically that means checking if its dependencies
are present in the environment, but can also control conditional use.

## Matching

When a pydantic model is instantiated and an `NDArray` is to be validated, 
{meth}`.Interface.match` first, uh, finds the matching interface.

Each interface must define a {meth}`.Interface.check` class that accepts the
array to be validated and returns whether it can be used. Interfaces can 
have any `check`ing logic they want, and so can eg. determine if a path 
is a particular type of file, but should return quickly and do little work 
since they are called frequently.

Validation fails if an argument doesn't match any interface.

```{note}
The {class}`.NumpyInterface` is special cased and is only checked if 
no other interface matches. It attempts to cast the input argument to a
{class}`numpy.ndarray` to see if it is arraylike, and since many 
lazy-loaded array libraries will attempt to load the whole array into memory
when cast to an `ndarray`, we only try as a last resort. 
```

## Validation

Validation is a chain of lifecycle methods, each of which can be overridden
for interfaces to implement custom behavior that matches the array format.

{meth}`.Interface.validate` calls the following methods, in order:

A method to deserialize the array dumped with a {func}`~pydantic.BaseModel.model_dump_json`
with `round_trip = True` (see [serialization](./serialization.md))
 
- {meth}`.Interface.deserialize`

An initial hook for modifying the input data before validation, eg.
if it needs to be coerced or wrapped in some proxy class. This method
should accept all and only the types specified in that interface's
{attr}`~.Interface.input_types`.

- {meth}`.Interface.before_validation`

A cluster of methods for validating dtype.
Separating these methods allow for array formats that store dtype information
in a nonstandard attribute, require additional coercion, or for implementing
custom exception handlers or rescuers.
Check the method signatures and return types
when overriding and the docstrings for details.

- {meth}`.Interface.get_dtype`
- {meth}`.Interface.validate_dtype`
- {meth}`.Interface.raise_for_dtype`

A halftime hook for modifying the array or bailing early between validation phases.

- {meth}`.Interface.after_validate_dtype`

A cluster of methods for validating shape, similar to the dtype cluster.

- {meth}`.Interface.get_shape`
- {meth}`.Interface.validate_shape`
- {meth}`.Interface.raise_for_shape`

A final hook for modifying the array before passing it to be assigned to the field.
This method should return an object matching the interface's {attr}`~.Interface.return_type`.
- {meth}`.Interface.after_validation`

## Diagram

```{todo}
Sorry this is unreadable, need to recall how to change the theme for 
generated mermaid diagrams but it is very late and i want to push this.
```

```{mermaid}
flowchart LR
    classDef data fill:#2b8cee,color:#ffffff;
    classDef X fill:transparent,border:none,color:#ff0000;

    input

    subgraph Interface
    match
    end

    subgraph Numpy
    numpy_check["check"]
    end

    subgraph Dask
    direction TB
    
    dask_check["check"]

    subgraph Validation
    direction TB
    
    before_validation --> validate_dtype
    validate_dtype --> validate_shape
    validate_shape --> after_validation
    end
    
    dask_check --> Validation

    end

    subgraph Zarr
    zarr_check["check"]
    end

    subgraph Model
    output
    end

    zarr_x["X"]
    numpy_x["X"]

    input --> match
    match --> numpy_check
    match --> zarr_check
    match --> Dask
    zarr_check --> zarr_x
    numpy_check --> numpy_x

    Validation --> Model

    class input data
    class output data
    class zarr_x X
    class numpy_x X
``` 