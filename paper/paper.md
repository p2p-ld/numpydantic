---
title: "Numpydantic: Static and dynamic array constraint validation for data standards and working programmers"
date: 27 May 2026
authors:
  - name: Jonny L. Saunders
    affiliation: '1'
    orcid: "0000-0003-0545-5066"
  - name: Daniel Aharoni
    affiliation: '1'
affiliations:
  - index: 1
    name: University of California, Los Angeles. Department of Neurology.
    ror: "046rm7j60"
bibliography: paper.bib
---

# Summary

Numpydantic is a minimalistic Python package that provides abstract types and a syntax for specifying constraint n-dimensional arrays.
Its constraints can both be validated at runtime and statically using mypy.
Rather than being bound to a single array framework,
it is generic array specification with an extensible interface system
that currently supports numpy, hdf5, zarr, and video files.
When used with pydantic, its NDArray type provides support for composing multiple arrays into validating data models,
including generation of detailed JSON schema,
roundtrip serialization,
and lazy proxies for on-disk arrays.
A mypy plugin enriches standard array constructors with dtype and shape,
and extends the quality of life improvements from static type analysis to the world of array programming. 

# Statement of need

The shape and dtype of an n-dimensional array are an essential part of its definition as a type.
Functions that can accept an array with any shape or dtype are the exception rather than the rule in scientific programming - 
arrays are rarely formless bundles of numbers, 
they are "video frames," "timeseries," "training and testing data," etc.
where the orientation, shape, and dtype are an intrinsic part of that identity.
In Python this information lives only in docstrings, 
is invisible to type checkers,
and requires large amounts of repetitive boilerplate to validate.
Scientific data standards that use n-dimensional arrays must specify array constraints as part of their schemas,
but the implementation of those constraints often bind the abstract form of a schema to a concrete set of tools and serialization formats.
Researchers who work with n-dimensional arrays have a rich variety of static analysis tooling to assist with software development,
but still struggle with keeping track of the hundreds of intermediate array forms that flow through their code.

The need for numpydantic then has three major parts:
- Need to be able to statically annotate array constraints to check code correctness during development
- Need to be able to dynamically validate data against array constraints at runtime
- Need to be able to declare schemas for arrays that are decoupled from a single array backend.

# State of the field

Python's type annotation system has evolved dramatically since its introduction in python 3.5 in 2014 [@vanrossumPEP484Type2014, @pythonsoftwarefoundationTypingPEPs].
Typing arrays is still an evolving challenge, 
motivating basic changes to the type system like variadic generics for shape specifications in PEP 646 [@mendozaPEP646Variadic2020].
Numpy introduced type annotations for its arrays in v1.20.0 in 2021 [@NumPy1200Release2021] from prior work in `numpy-stubs` [@NumpyNumpystubs2017], with steady improvements over time. Numpy's types allow specifying the number of dimensions and a dtype, but not common needs like explicit dimension sizes, size ranges, etc., and its constructors. Numpydantic extends previous work attempting to provide runtime-checkable type,
vendoring `nptyping` [@hagenaarsNptyping2019] and extending and updating its syntax for new features of the python type system. 

Other array frameworks like Dask [@daskdevelopmentteamDaskLibraryDynamic2016]
and Zarr [@milesZarrdevelopersZarrpythonV2402020]
are either untyped or allow typing only for dtype.

Dataclasses [@smithPEP557Data2017] and dataclass-like packages use type annotations to create intelligible data structures, an avenue to escape the wild free-for-all of passing around un-annotated `dict`s and classes.
Pydantic [@colvinPydanticValidation2026] is the most widely used library in python for declaring runtime-validating dataclass-like models,
but it lacks builtin support for arrays.

The closest sibling to numpydantic is likely [pandera](https://pandera.readthedocs.io) [@bantilanPanderaStatisticalData2020],
which provides schemas for dataframes across multiple backends (and, recently, complex multi-arrays from [xarray](https://docs.xarray.dev)). 
We are not aware of other projects that provide array specifications and validation across multiple array backends[^contactus].

[^contactus]: If you maintain an array spec library for python, please contact us and we will add you to our documentation as related projects!

# Software design

Numpydantic is designed to be a dependency you don't have to think about -
it has two required dependencies, numpy and pydantic, which are already likely present in any package that wants to use it.
It can be used with or without pydantic, its annotations can be used in existing code in the typing layer for functions and variables without modifying any code.

Support for multiple array frameworks is implemented as a set of [interface classes](https://numpydantic.readthedocs.io/en/latest/interfaces.html),
where each framework overrides the methods needed to match input to an interface,
extract shape and dtype,
serialize and deserialize json,
among other features.

## Pydantic Models

At its simplest, numpydantic does this:

```python
import numpy as np
from pydantic import BaseModel
from numpydantic import NDArray, Shape

class MyModel(BaseModel):
    array: NDArray[Shape["* x, * y, 3 rgb, * t"], np.uint8]
    
# numpy
MyModel(array=np.zeros((1080, 1920, 3, 100), dtype=np.uint8))
# dask
MyModel(array=dask.array.zeros((3, 4, 5), dtype=np.uint8))
# hdf5 datasets
MyModel(array=('data.h5', '/nested/dataset'))
# zarr arrays
MyModel(array=zarr.zeros((3,4,5), dtype=np.uint8))
MyModel(array=('data.zarr', '/nested/dataset'))
# video files
MyModel(array="data.mp4")
```

When used in pydantic models, numpydantic provides runtime validation for array specifications,
as well as access to on-disk arrays by reference as in the HDF5, zarr, and video examples above.

By default, an array from any supported array backend can be used (as long as it matches the specification),
but an explicit array backend can be specified using an annotated type

```python
class MyModel(BaseModel):
    array: Annotated[np.ndarray, NDArraySchema(Shape(1, 2, 3), np.uint8)]
```

Numpydantic constructs a valid JSON schema for its annotations,
making them portable across languages,
and usable to accept array data via web APIs via, e.g. [fastAPI](https://fastapi.tiangolo.com/).
For a simple example, for a `float16` array whose first dimension is length 1,
the second can be between 2 and 5, the following JSON schema is produced (abbreviated for clarity):

```python
class MyModel(BaseModel):
    array: NDArray[Shape[1, "2-5"], np.float16]
MyModel.model_json_schema()
```

```json
{"properties": {"array": {
  "dtype": "numpy.float16",
  "items": {
    "items": {
      "maximum": 65504.0,
      "minimum": -65504.0,
      "type": "integer"
    },
    "maxItems": 5,
    "minItems": 2,
    "type": "array"
  },
  "maxItems": 1,
  "minItems": 1,
  "type": "array"
}}}
```

Numpydantic supports roundtripping array data to and from JSON,
including any metadata needed to reconstruct an array,
and allowing on-disk arrays to be dumped as a reference rather than loading and serializing the whole array.

## Standalone Use

Numpydantic can be used without pydantic to annotate normal functions and classes!

```python
def some_analysis(array: NDArray[Shape[1, 2, 3]]) -> NDArray[Shape[2, 4, 6]]:
    ...
```

These annotations can also be validated at runtime using a tool like [beartype](https://beartype.readthedocs.io/) [@curryBeartypeBeartype2020].

It can also be used as a runtime type for instance checking,
as well as ad-hoc validation

```python
>>> Grayscale = NDArray[Shape["*", "*"], np.uint8]
>>> array = np.zeros((100, 100), np.uint8)
>>> isinstance(np.zeros((100, 100), np.uint8), Grayscale)
True
>>> isinstance(np.zeros((100,)), Grayscale)
False
>>> Grayscale(np.zeros((100,), np.uint8))
ShapeError: Invalid shape! expected shape ['*', '*'], got shape (100,)
```

### Mypy plugin

A mypy plugin supports static analysis of numpydantic's specification syntax,
enriching array constructors with shape and dtype information
for any supported array backend.

```python
reveal_type(np.zeros((3, 4, 5), dtype=np.uint8))
reveal_type(zarr.zeros((3, 4, 5), another=int, dtype=np.uint8))
```

Without the plugin:

```
Revealed type is "numpy.ndarray[tuple[int, int, int], numpy.uint8]"
Revealed type is "Any"
```

With the plugin:

```
Revealed type is "numpy.ndarray[tuple[Literal[3], Literal[4], Literal[5]], numpy.uint8]"
Revealed type is "numpy.ndarray[tuple[Literal[3], Literal[4], Literal[5]], numpy.uint8]"
```

By annotating analysis and i/o functions with correct type information,
researchers can avoid tedious bugs from misshaped or typed arrays,
and use the runtime types for systematic validation without boilerplate.

# Research impact statement

Numpydantic was developed as part of the design of [LinkML arrays](https://linkml.io/linkml/schemas/arrays.html) [@moxonLinkMLOpenData2026].
Briefly, LinkML is an accessible, expressive schema language that allows a schema defined once to be translated into a number of schema and programming languages like OWL, JSON Schema, SQL DDL, etc. 
LinkML is widely used for linked data schemas and ontologies,
and is increasingly a target for adoption for scientific data formats.
Support for n-dimensional arrays is a critical part of using LinkML for scientific data,
and numpydantic is the core implementation for LinkML in Python.

Numpydantic was also developed as part of an effort to rearchitect Neurodata Without Borders (NWB) [@rubelNeurodataBordersEcosystem2022].
NWB uses HDMF [@trittHDMFHierarchicalData2019] as its schema language,
which was developed out of necessity,
as most schema languages do not support array specifications.
HDMF and NWB are both tightly coupled to HDF5 as a serialization format,
which makes metadata indexing and partial use of large datasets extraordinarily costly - 
to check a single scalar metadata attribute,
one needs to download the entire dataset,
which for neuroscientific data often means hundreds of gigabytes.
With NWB backed by linkml arrays and numpydantic
the specification becomes decoupled from the serialization format,
allowing array backends to be freely swapped among any supported by numpydantic,
and separating lightweight metadata from heavier raw array data.
By decoupling schema, programmatic APIs, and serialization formats
with LinkML as a *lingua franca,* we are working towards a true ecosystem of FAIR data [@wilkinsonFAIRGuidingPrinciples2016]
where data standards only need to define the schema layer and get the rest for free.

# Limitations & Future Directions

- We are working towards [v2.0 of numpydantic](https://numpydantic.readthedocs.io/en/latest/development.html#versioning),
  where we will remove the string-based syntax borrowed from `nptyping`
  and replace it with modern Python typing features.
- Because this isn't done yet,
  non-mypy static type checkers have trouble checking some forms of `NDArray` types.
  We have introduced [various ways of resolving this problem](https://numpydantic.readthedocs.io/en/latest/syntax.html#type-checker-compatibility),
  like specifying shape arguments as literals,
  and the `NDArraySchema` class which type checkers ignore since it is in an `Annotated` type.
- We want to extend the specification syntax to include more advanced array features,
  like chunking, memory order, and so on.
  There is not a strong demand for these features yet,
  but when there is we will implement them.
- Support for TypeVars that can indicate that a given axis can be any size,
  but all shapes within a given scope must be the same size is important
  for many scientific applications 
  (e.g. a timestamps array must be the same length as some video array).
  We are [actively working on this](https://github.com/p2p-ld/numpydantic/issues/70)
  both in our mypy plugin as well as [upstream in mypy itself](https://github.com/python/mypy/issues/3345#issuecomment-4664544812)

# AI usage disclosure

No "AI" or LLMs were used at any time during development, nor to produce any part of this submission.

# Acknowledgements

Thanks to Ryan Ly, Chris Mungall, Oliver Rübel, Wouter-Michiel Vierdag, and Ben Dichter for co-designing the LinkML Arrays specification
which gave shape to the initial spec; 
to Antônio Camargo for setting up the conda forge feedstock;
to all co-contributors past and present;
and to Rumbly Tumbly Lawnmower for being the light of my life and keeping my lap warm as I worked.

# References