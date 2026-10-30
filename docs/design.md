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

For the numpy+pydantic case specifically, see also [pydantic-numpy](https://github.com/caniko/pydantic-numpy)

(design_challenges)=
## Challenges

The Python type annotation system is weird and not like the rest of Python!
Similarly, Pydantic 2's core_schema system is wonderful but still has a few mysteries
lurking under the documented surface.
This package does the work of plugging them in
together to make some kind of type validation frankenstein.

The first problem is that type annotations are evaluated statically by typecheckers like, mypy, etc. 
The type can't have any dynamic behavior because type checkers do not execute the code at all, 
that's what static type annotations are!
So unlike the rest of python, "real" and "correct" type annotations must follow the typing spec exactly,
and don't really have a path for customizing their behavior except by 
[writing a plugin for a typechecker](./typecheckers.md).
Since there are some typecheckers like pyright that refuse to allow customization out of principle,
we are *sort of* SoL if we want to do things like support type transformations 
across function calls and class instantiations,
or have a more humane annotation than 
`NDArray[tuple[Annotated[Literal[1920], Name("width")], Annotated[Literal[1080], Name("height")]], np.uint8]`
to annotate, e.g. a frame from an HD video.

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

As of 1.9.0 - we have added [an experimental mypy plugin](mypy-plugin)
```

Pydantic builds a core schema for each annotation once, when a model class is defined,
and validates with that schema afterwards. A validator function in a core schema
is only called with the value and a `ValidationInfo`,
which says which field is being validated but not what its annotation was.
So the `shape` and `dtype` of `NDArray[shape, dtype]` are read when the schema is built,
from the alias's type arguments, and bound into the validator with a
{func}`closure <numpydantic.validation.array.get_validate_interface>`,
which applies them to the array when it's validated.

Numpydantic started out using `nptyping`'s syntax for array specifications.
It was a longstanding answer to the desire for more granular array type annotations,
but it was developed before some key developments in python and its typing system,
and is no longer actively maintained.
Shapes are now ordinary type expressions: to type checkers, `Shape` is `tuple`,
and at runtime `Shape[...]` returns the equivalent `tuple[...]` (see [syntax](syntax.md)).
`nptyping`'s string form, like `Shape["3, 4"]`, is deprecated, and will be removed in 3.0.
`nptyping` is no longer vendored, so the only required dependencies
are {mod}`numpy` and {mod}`pydantic`.

Numpydantic is *not* an array library, but a tool that allows you to use existing
array libraries with pydantic. It tries to be a transparent passthrough to 
whatever library you are using, adding only minimal convenience classes to
make array usage roughly uniform across array libraries, but otherwise exposing
as much of the functionality of the library as possible.

It is designed to be something that you don't have
to think too carefully about before adding it as a dependency - it is simple, 
clean, unsurprising, well tested, and has three required dependencies.