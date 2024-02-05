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



 
```{toctree}
:maxdepth: 2
:caption: Contents
:hidden: true

overview
ndarray
linkml
hooks
todo
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden: true

api/index
api/ndarray
api/proxy
api/linkml/index
api/maps
api/monkeypatch

```

