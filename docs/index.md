# numpydantic

Type and shape validation and serialization for numpy arrays in pydantic models

This package was picked out of [nwb-linkml](https://github.com/p2p-ld/nwb-linkml/), a 
translation of the [NWB](https://www.nwb.org/) schema language and data format to
linkML and pydantic models. It's in a hurried and limited form to make it
available for a LinkML hackathon, but will be matured as part of `nwb-linkml` development
as the primary place this logic exists.

It does two primary things:
- **Provide types** - Annotations (based on [npytyping](https://github.com/ramonhagenaars/nptyping))
  for specifying numpy arrays in pydantic models, and
- **Generate models from LinkML** - extend the LinkML pydantic generator to create models that 
  that use the [linkml-arrays](https://github.com/linkml/linkml-arrays) syntax

## Overview

The Python type annotation system is weird and not like the rest of Python! 
(at least until [PEP 0649](https://peps.python.org/pep-0649/) gets mainlined).
Similarly, Pydantic 2's core_schema system is wonderful but still relatively poorly
documented for custom types! This package does the work of plugging them in
together to make some kind of type validation frankenstein.

The first problem is that type annotations are evaluated statically by python, mypy,
etc. This means you can't use typical python syntax for declaring types - it has to
be present at the time `__new__` is called, rather than `__init__`. 

- pydantic schema
- validation
- serialization
- lazy loading
- compression


```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

hooks
```

