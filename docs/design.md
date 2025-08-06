# Design

## Why do this?

We want to bring the tidiness of modeling data with pydantic to the universe of
software that uses arrays - particularly formats and packages that need to be very 
particular about what *kind* of arrays they are able to handle or match a specific schema.

To support a new generation of data formats and data analysis libraries that can
model the *structure* of data independently from its *implementation,* we made 
numpydantic as a bridge between abstract schemas and programmatic use.

The closest prior work is likely [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping),
but its support for multiple array libraries was backed into from its initial 
design as a `jax` specification package, and so its extensibility and readability is
relatively low. Its `Dtype[ArrayClass, "{shape_expression}"]` syntax is not well 
suited for modeling arrays intended to be general across implementations, and 
makes it challenging to adapt to pydantic's schema generation system.

(design_challenges)=
## Challenges

The Python type annotation system is weird and not like the rest of Python! 
(at least until [PEP 0649](https://peps.python.org/pep-0649/) gets mainlined).
Similarly, Pydantic 2's core_schema system is wonderful but still has a few mysteries
lurking under the documented surface.
This package does the work of plugging them in
together to make some kind of type validation frankenstein.

The first problem is that type annotations are evaluated statically by python, mypy,
etc. This means you can't use typical python syntax for declaring types - it has to
be present at the time `__new__` is called, rather than `__init__`. So  

Different implementations of arrays behave differently! HDF5 files need to be carefully
opened and closed to avoid corruption, video files don't typically allow normal array
slicing operations, and only some array libraries support lazy loading of arrays on disk.

We can't anticipate all the possible array libraries that exist now or in the future,
so it has to be possible to extend support to them without needing to go through
a potentially lengthy contribution process.

## Strategy

Numpydantic uses {class}`~numpydantic.NDArray` as an abstract specification of 
an array that uses one of several [interface](interfaces.md) classes to validate
and interact with an array. These interface classes will set the instance attribute
either as the passed array itself, or a transparent proxy class (eg. 
{class}`~numpydantic.interface.hdf5.H5Proxy`) in the case that the native array format
doesn't support numpy-like array operations out of the box.

The `interface` validation process thus often transforms the type of the passed array -
eg. when specifying an array in an HDF5 file, one will pass some reference to
a `Path` and the location of a dataset within that file, but the returned value from the
interface validator will be an {class}`~numpydantic.interface.hdf5.H5Proxy` 
to the dataset. This confuses python's static type checker and IDE integrations like
pylance/pyright/mypy, which naively expect the type to literally be an
{class}`~numpydantic.NDArray` instance. To address this, numpydantic generates a `.pyi`
stub file on import (see {mod}`numpydantic.meta` ) that declares the type of `NDArray`
as the union of all {attr}`.Interface.return_types` .

```{todo}
To better support static type hinting and inspection (ie. so the type checker
is not only aware of the union of all `return_types`, but the specific array
type that was passed on model instantiation, as well as potentially
do shape and dtype checks during type checking (eg. so a wrongly shaped or dtyped 
array assignment will be highlighted as wrong), we will be exploring adding 
mypy/pylance/pyright hooks for dynamic type evaluation.
```

Since type annotations are static, each `NDArray[]` usage effectively creates a new
class. The `shape` and `dtype` specifications are thus not available at the time
that the validation is performed (see how [pydantic handles Annotated types](https://github.com/pydantic/pydantic/blob/87adc65888ce54ef4314ef874f7ecba52f129f84/pydantic/_internal/_generate_schema.py#L1788)
at the time that the class definition is evaluated by generating pydantic "core schemas", 
which are passed to the rust `pydantic_core` for fast validation, which can't be 
done with python-based validation functions). The validation function for each
`NDArray` pseudo-subclass is a {func}`closure <numpydantic.schema.get_validate_interface>` 
that uses the *class declaration*-timed `shape` and `dtype` annotations with the
*instantiation*-timed array object to find the matching validator interface and apply it.

We are initially adopting `nptyping`'s syntax for array specification. It is a longstanding
answer to the desire for more granular array type annotations, but it also was 
developed before some key developments in python and its typing system, and is 
no longer actively maintained. We make some minor modifications to its 
{mod}`~numpydantic.dtype` specification (eg. to allow builtin python types like `int`
and `float`), but any existing `nptyping` annotations can be used as-is with
`numpydantic`. In [v2.*](todo.md#v2) we will be reimplementing it, as well as 
making an extended syntax for shape and dtype specifications, so that the 
only required dependencies are {mod}`numpy` and {mod}`pydantic`. This will also
let us better hook into pydantic 2's use of `Annotated`, eliminating some
of the complexity in how specification information is passed to the validators.

Numpydantic is *not* an array library, but a tool that allows you to use existing
array libraries with pydantic. It tries to be a transparent passthrough to 
whatever library you are using, adding only minimal convenience classes to
make array usage roughly uniform across array libraries, but otherwise exposing
as much of the functionality of the library as possible.

It is designed to be something that you don't have
to think too carefully about before adding it as a dependency - it is simple, 
clean, unsurprising, well tested, and has three required dependencies.