---
title: "Numpydantic: Static and dynamic array constraint validation for data standards and working programmers"
date: 27 May 2026
authors:
  - name: Jonny L. Saunders
    affiliation: '1, 2'
    orcid: "0000-0003-0545-5066"
  - name: Daniel Aharoni
    affiliation: '1'
    orcid: "0000-0003-4931-8514"
affiliations:
  - index: 1
    name: University of California, Los Angeles. Department of Neurology.
    ror: "046rm7j60"
  - index: 2
    name: Peertech Global Cyberindustrial Concern LLC, LLC.
bibliography: paper.bib
---

# Summary

Numpydantic is a minimalistic Python package that provides abstract types and a syntax for specifying constrained n-dimensional arrays.
Its constraints can be validated both dynamically at runtime and statically using mypy.
Rather than being bound to a single array framework,
numpydantic is a generic array specification with an extensible interface system
that currently supports numpy, hdf5, zarr, and video files.
Designed to expose arrays to `pydantic`'s validation framework, 
its `NDArray` type allows researchers to compose multiple arrays into validating data models,
including generation of JSON schema,
roundtrip serialization,
and lazy proxies for on-disk arrays.
A mypy plugin extends the quality of life improvements from static type analysis to the world of array programming.
Numpydantic is part of a broader effort to lower barriers to structured data and provide a path to true cross-standard interoperability:
decoupling schemas, implementations, and serialization formats for formal data standards,
and providing simple tools responsive to the needs of everyday research programming.

# Statement of need

Arrays are rarely formless bundles of numbers, 
they are "videos," "time series," "training and testing data," etc.
where attributes like orientation, shape, and dtype are an intrinsic part of their definition as a type.
Accordingly, outside of array frameworks themselves,
array code rarely accepts arbitrary arrays.
In Python this information lives only in docstrings, 
is invisible to type checkers,
and requires large amounts of repetitive boilerplate to validate.

The lofty ambitions of open data require standards and formats [@wilkinsonFAIRGuidingPrinciples2016]. 
Scientific data standards must specify array constraints,
but the implementation of those constraints often binds the abstract schema to a single array framework and serialization format.
Researchers have a rich set of static analysis tools to assist with programming,
but still struggle with the hundreds of intermediate arrays that flow through their code.

This tension creates two unhappy local minima:
on one side, data standards are often brittle and difficult to integrate into the code used for real-world analysis, 
relegated to a conversion stage after the work is done that can be lossy and labor-intensive.
On the other, labs run on ad-hoc, local data structures that make even intra-lab collaboration difficult.

In the middle is a gap in tooling that responds to both categories of needs:

- Strict, validating, metadata enriched schemas with loose coupling to storage formats and programming interfaces
- Simple, low-overhead structure that can embed in existing code and support researcher-developer experience.

# State of the field

Python's type annotation system has evolved dramatically since its introduction in 2014 [@vanrossumPEP484Type2014; @pythonsoftwarefoundationTypingPEPs].
Typing arrays is an ongoing challenge, 
motivating type system features like variadic generics for shape specification [@mendozaPEP646Variadic2020].
Numpy [@harrisArrayProgrammingNumPy2020] introduced type annotations for its arrays in 2021 [@NumPy1200Release2021], 
with steady improvements over time[^numtype]. 
Numpy's types allow specifying shape and dtype,
but not extended expressions like explicit dimension sizes, size ranges, etc. 
Numpydantic extends previous work on runtime-checkable types,
vendoring `nptyping` [@hagenaarsNptyping2019][^unmaintained] and extending and updating its syntax. 

Other array frameworks like Dask [@daskdevelopmentteamDaskLibraryDynamic2016]
and Zarr [@milesZarrdevelopersZarrpythonV2402020]
are either untyped or allow typing only for dtype.

Dataclasses [@smithPEP557Data2017] use type annotations to create intelligible data structures,
but the annotations are unvalidated and used solely in static analysis.
Pydantic [@colvinPydanticValidation2026] is the most widely used library in Python for declaring runtime-validating dataclass-like models,
but it lacks builtin support for arrays.

The closest sibling to numpydantic is likely [pandera](https://pandera.readthedocs.io) [@bantilanPanderaStatisticalData2020],
which provides schemas for dataframes across multiple backends (and, recently, complex multi-arrays from [xarray](https://docs.xarray.dev)). 
We are not aware of other projects that provide array specifications and validation across multiple array backends[^contactus].

[^numtype]: See also the predecessor `numpy-stubs` [@numpyNumpystubs2017] and the experimental [numtype](https://github.com/numpy/numtype) [@numpyNumtype2025].
[^unmaintained]: As of writing, `nptyping` is unmaintained, without update since February 2023.
[^contactus]: If you maintain an array spec library for Python, please contact us and we will link to you from our docs!

# Software design

Numpydantic is designed to be a dependency you don't have to think about -
it has two required dependencies, numpy and pydantic, which are already likely present in any package that wants to use it.
It can be used with or without pydantic:
its annotations can be used in the typing layer without modifying existing runtime code.

Support for multiple array frameworks is implemented as a set of [interface classes](https://numpydantic.readthedocs.io/en/latest/interfaces.html),
where each framework overrides the methods needed to match input to an interface,
extract shape and dtype,
serialize and deserialize json,
among other features.

\pagebreak

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
MyModel(array=dask.array.zeros((1080, 1920, 3, 100), dtype=np.uint8))
# hdf5
MyModel(array=('data.h5', '/nested/dataset'))
# zarr
MyModel(array=zarr.zeros((1080, 1920, 3, 100), dtype=np.uint8))
MyModel(array=('data.zarr', '/nested/dataset'))
# video
MyModel(array="data.mp4")
```

When used in pydantic models, numpydantic provides runtime validation for array specifications,
as well as access to on-disk arrays by reference.

By default, an array from any supported array backend can be used (as long as it matches the specification),
but an explicit array backend can be specified using an annotated type

```python
class MyModel(BaseModel):
    array: Annotated[np.ndarray, NDArraySchema(Shape(1, 2, 3), np.uint8)]
```

Numpydantic constructs a valid JSON schema for its annotations,
making them portable across languages,
and usable to accept array data via web APIs via, e.g. [fastAPI](https://fastapi.tiangolo.com/).

This simple example:

```python
class MyModel(BaseModel):
    array: NDArray[Shape[2, "2-5"], np.float16]
MyModel.model_json_schema()
```

produces JSON Schema[^abbreviated]:

```json
{ "properties": { "array": {
  "dtype": "numpy.float16",
  "items": {
    "items": {
      "minimum": -65504.0, "maximum": 65504.0,
      "type": "integer"
    },
    "minItems": 2, "maxItems": 5,
    "type": "array"
  },
  "minItems": 2, "maxItems": 2,
  "type": "array"
}}}
```

\pagebreak

Numpydantic supports roundtripping array data to and from JSON,
including any metadata needed to reconstruct an array,
with on-disk arrays optionally dumped as a reference rather than loading and serializing the whole array.

e.g. for a dataset with scalar metadata alongside array data split between in-memory and on-disk arrays,
as would be common in both data standards and unstructed lab code:

```python
MyDataset(
  name   = "My Dataset",
  time   = datetime.now(UTC),
  scores = np.zeros((2, 2)),
  raw    = ZarrArrayPath("./data.zarr", "/dataset"),
  video  = Path('./video.mp4')
).model_dump_json(round_trip=True)
```

A lightweight metadata descriptor can be dumped to JSON and reloaded later:

```json
{ "name": "My Dataset",
  "time": "2026-06-11T02:17:53.973174Z",
  "scores": {
    "type": "numpy", "dtype": "float64", "shape": [ 2, 2 ],
    "value": [ [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]
  },
  "raw": {
    "type": "zarr",      "dtype": "float64",
    "file": "data.zarr", "path": "/dataset",
    "info": {
      "Shape": "(2000, 2000)",
      "hexdigest": "86c96d94e76c3b0f210f5a0bf2d8811d1ffd883d"
    }
  },
  "video": { "type": "video", "file": "video.mp4" }
}
```

[^abbreviated]: All examples are abbreviated and formatted for clarity.

## Standalone Use

Numpydantic can be used without pydantic to annotate normal functions and classes!

```python
def some_analysis(array: NDArray[Shape[1, 2, 3]]) -> NDArray[Shape[2, 4, 6]]:
    ...
```

These annotations can also be validated at runtime without pydantic using a tool like [beartype](https://beartype.readthedocs.io/) [@curryBeartype2020].

`NDArray` can also be used as a runtime type for instance checking and ad-hoc validation

```python
>>> Grayscale = NDArray[Shape["*", "*"], np.uint8]
>>> array = np.zeros((100, 100), np.uint8)
>>> isinstance(np.zeros((100, 100), np.uint8), Grayscale)
True
>>> isinstance(np.zeros((100,)), Grayscale)
False
>>> Grayscale(np.zeros((100,), np.uint8))
ShapeError: Invalid shape! expected shape ['*', '*'], got shape (100,)
>>> Grayscale(np.zeros((100, 100), np.float64))
DtypeError: Invalid dtype! expected <class 'numpy.uint8'>, got float64
```

### Mypy plugin

A mypy plugin supports static analysis of numpydantic's specification syntax,
enriching array constructors with shape and dtype information
for any supported array backend.

```python
reveal_type(np.zeros((3, 4), dtype=np.uint8))
reveal_type(zarr.zeros((3, 4), dtype=np.uint8))
```

Without the plugin:

```
Revealed type is "numpy.ndarray[tuple[int, int], numpy.uint8]"
Revealed type is "Any"
```

With the plugin:

```
Revealed type is "numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.uint8]"
Revealed type is "numpy.ndarray[tuple[Literal[3], Literal[4]], numpy.uint8]"
```

By annotating analysis and i/o functions with type information,
researchers can avoid tedious bugs from misshaped or mistyped arrays,
and use the runtime types for reusable, minimal boilerplate validation.

```python
def analyze_grayscale(frame: NDArray[Shape[1080, 1920], np.uint8]):
    # ...

analyze_grayscale(np.zeros((1080, 1920, 3), dtype=np.uint8))
```

```
error: Argument "frame" to "analyze_grayscale" has incompatible type 
  "ndarray[tuple[Literal[1080], Literal[1920], Literal[3]], numpy.uint8]"; 
  expected "ndarray[tuple[Literal[1080], Literal[1920]], np.uint8]" 
```

# Research impact statement

Numpydantic was developed as part of [LinkML arrays](https://linkml.io/linkml/schemas/arrays.html) [@moxonLinkMLOpenData2026].
Briefly, LinkML is an accessible, expressive schema language that allows a schema defined once to be translated into a number of schema and programming languages like OWL, JSON Schema, SQL DDL, etc. 
LinkML is widely used for linked data schemas and ontologies,
and is increasingly an adoption target for scientific data formats.
Support for n-dimensional arrays is a critical part of using LinkML for scientific data,
and numpydantic is the core array implementation for LinkML in Python.

Numpydantic was also developed as part of an effort to rearchitect Neurodata Without Borders (NWB) [@rubelNeurodataBordersEcosystem2022],
the de-facto data standard for systems neuroscience ([`nwb-linkml`](https://github.com/p2p-ld/nwb-linkml)).
NWB uses HDMF [@trittHDMFHierarchicalData2019] as its schema language,
which was developed out of necessity
as most schema languages do not support arrays.
HDMF and NWB are both tightly coupled to HDF5,
which makes indexing and partial use of datasets extraordinarily costly - 
to read a scalar metadata attribute,
one needs to download the entire dataset,
which for neuroscientific data often means hundreds of gigabytes[^lindi].
Backing NWB with linkml arrays and numpydantic
allows array backends to be freely swapped,
and separates metadata from heavier raw array data.
By decoupling schema, programmatic APIs, and serialization formats
with LinkML as a *lingua franca,* we are working towards a true ecosystem of FAIR data [@wilkinsonFAIRGuidingPrinciples2016]
where data standards only need to define the schema layer and get the rest for free.

[^lindi]: See also [LINDI](https://github.com/NeurodataWithoutBorders/lindi) [@neurodatawithoutbordersLindi2026],
  an overlay for addressing streaming data needs with HTTP range requests.

# Limitations & Future Directions

- We are working on [v2.0 of numpydantic](https://numpydantic.readthedocs.io/en/latest/development.html#versioning),
  where we will replace the string-based syntax borrowed from `nptyping`
  with modern Python typing features.
- Until then,
  non-mypy static type checkers have trouble checking some forms of `NDArray` types.
  There are [several intermediate solutions](https://numpydantic.readthedocs.io/en/latest/syntax.html#type-checker-compatibility),
  like specifying shape arguments as literals,
  and the `NDArraySchema` class.
- We will be extending the specification syntax to include more advanced array features,
  like chunking and memory order.
- Support for TypeVars that can indicate that a given axis can be any size,
  but all axes within a given scope must be the same size is important
  for many scientific applications 
  (e.g. a timestamps array must be the same length as a video.
  We are [actively working on this](https://github.com/p2p-ld/numpydantic/issues/70)
  in our mypy plugin and [upstream in mypy itself](https://github.com/python/mypy/issues/3345#issuecomment-4664544812)

# AI usage disclosure

No "AI" or LLMs were used at any time during development, nor to produce any part of this submission.

# Acknowledgements

Thanks to Ryan Ly, Chris Mungall, Oliver Rübel, Wouter-Michiel Vierdag, Ben Dichter, and John-Marc Chandonia for co-designing the LinkML Arrays specification
which gave shape to the initial spec; 
to Antônio Camargo for setting up the conda forge feedstock;
to all collaborators past and present;
and to Rumbly Tumbly Lawnmower for being the light of my life and keeping my lap warm as I worked.

We are grateful for the funding that supported this work: 
NIH Director's New Innovator Award, DP2MH129986.

# References