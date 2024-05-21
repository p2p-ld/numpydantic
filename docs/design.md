# Design

## Why do this?

We want to bring the tidyness of modeling data with pydantic to the universe of
software that uses arrays - particularly formats and packages that need to be very 
particular about what *kind* of arrays they are able to handle or match a specific schema.

## Challenges

The Python type annotation system is weird and not like the rest of Python! 
(at least until [PEP 0649](https://peps.python.org/pep-0649/) gets mainlined).
Similarly, Pydantic 2's core_schema system is wonderful but still has a few mysteries
lurking under the documented surface.
This package does the work of plugging them in
together to make some kind of type validation frankenstein.

The first problem is that type annotations are evaluated statically by python, mypy,
etc. This means you can't use typical python syntax for declaring types - it has to
be present at the time `__new__` is called, rather than `__init__`. 

- pydantic schema
- validation
- serialization
- lazy loading
- compression