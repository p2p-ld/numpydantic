# Design

## Why do this?

We want to bring the tidyness of modeling data with pydantic to the universe of
software that uses arrays - particularly formats and packages that need to be very 
particular about what *kind* of arrays they are able to handle or match a specific schema.

To support a new generation of data formats and data analysis libraries that can
model the *structure* of data independently from its *implementation,* we made 
numpydantic as a bridge between abstract schemas and programmatic use.

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

- type hinting
- nptyping syntax
- not trying to be an array library
- dtyping, mapping & schematization