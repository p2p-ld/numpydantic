---
file_format: mystnb
mystnb:
    output_stderr: remove
    render_text_lexer: python
    render_markdown_format: myst
myst:
    enable_extensions: ["colon_fence"]
---

# Serialization

## Python

In most cases, dumping to python should work as expected.

When a given array framework doesn't provide a tidy means of interacting
with it from python, we substitute a proxy class like {class}`.hdf5.H5Proxy`,
but aside from that numpydantic {class}`.NDArray` annotations
should be passthrough when using {func}`~pydantic.BaseModel.model_dump` .

## JSON

JSON is the ~ ♥ fun one ♥ ~

There isn't necessarily a single optimal way to represent all possible
arrays in JSON. The standard way that n-dimensional arrays are rendered
in json is as a list-of-lists (or array of arrays, in JSON parlance),
but that's almost never what is desirable, especially for large arrays.

### Normal Style[^normalstyle]

Lists-of-lists are the standard, however, so it is the default behavior
for all interfaces, and all interfaces must support it.

```{code-cell}
---
tags: [hide-cell]
---

from pathlib import Path
from pydantic import BaseModel
from numpydantic import NDArray, Shape
from numpydantic.interface.dask import DaskJsonDict
from numpydantic.interface.numpy import NumpyJsonDict
import numpy as np
import dask.array as da
import zarr
import json
from rich import print
from rich.console import Console

def print_json(string:str):
    data = json.loads(string)
    console = Console(width=74)
    console.print(data)
```

For our humble model:

```{code-cell}
class MyModel(BaseModel):
    array: NDArray
```

We should get the same thing for each interface:

```{code-cell}
model = MyModel(array=[[1,2],[3,4]])
print(model.model_dump_json())
```

```{code-cell}
model = MyModel(array=da.array([[1,2],[3,4]], dtype=int))
print(model.model_dump_json())
```

```{code-cell}
model = MyModel(array=zarr.array([[1,2],[3,4]], dtype=int))
print(model.model_dump_json())
```

```{code-cell}
model = MyModel(array="docs/data/test.avi")
print(model.model_dump_json())
```

(ok maybe not that last one, since the video reader still incorrectly
reads grayscale videos as BGR values for now, but you get the idea)

Since by default arrays are dumped into unadorned JSON arrays,
when they are re-validated, they will always be handled by the
{class}`.NumpyInterface`

```{code-cell}
dask_array = da.array([[1,2],[3,4]], dtype=int)
model = MyModel(array=dask_array)
type(model.array)
```

```{code-cell}
model_json = model.model_dump_json()
deserialized_model = MyModel.model_validate_json(model_json)
type(deserialized_model.array)
```

All information about `dtype` will be lost, and numbers will be either parsed
as `int` ({class}`numpy.int64`) or `float` ({class}`numpy.float64`)

## Roundtripping

To roundtrip make arrays round-trippable, use the `round_trip` argument
to {func}`~pydantic.BaseModel.model_dump_json`


```{code-cell}
print_json(model.model_dump_json(round_trip=True))
```

Each interface should[^notenforced] implement a dataclass that describes a
json-able roundtrip form (see {class}`.interface.JsonDict`).

That dataclass then has a {meth}`JsonDict.is_valid` method that checks
whether an incoming dict matches its schema

```{code-cell}
roundtrip_json = json.loads(model.model_dump_json(round_trip=True))['array']
DaskJsonDict.is_valid(roundtrip_json)
```

```{code-cell}
NumpyJsonDict.is_valid(roundtrip_json)
```

#### Controlling paths

When possible, the full content of the array is omitted in favor
of the path to the file that provided it.

```{code-cell}
model = MyModel(array="docs/data/test.avi")
print_json(model.model_dump_json(round_trip=True))
```

```{code-cell}
model = MyModel(array=("docs/data/test.h5", "/data"))
print_json(model.model_dump_json(round_trip=True))
```

You may notice the relative, rather than absolute paths.


We expect that when people are dumping data to json in this roundtripped
form that they are either working locally 
(e.g. transmitting an array specification across a socket in multiprocessing
or in a computing cluster),
or exporting to some directory structure of data, 
where they are making an index file that refers to datasets in a directory
as part of a data standard or vernacular format.

By default, numpydantic uses the current working directory as the root to find
paths relative to, but this can be controlled by the [`relative_to`](#relative_to)
context parameter:

For example if you're working on data in many subdirectories,
you might want to serialize relative to each of them:

```{code-cell}
print_json(
  model.model_dump_json(
    round_trip=True, 
    context={"relative_to": Path('./docs/data')}
  ))
```

Or in the other direction:

```{code-cell}
print_json(
  model.model_dump_json(
    round_trip=True, 
    context={"relative_to": Path('../')}
  ))
```

Or you might be working in some completely different place,
numpydantic will try and find the way from here to there as long as it exists,
even if it means traversing to the root of the readthedocs filesystem

```{code-cell}
print_json(
  model.model_dump_json(
    round_trip=True, 
    context={"relative_to": Path('/a/long/distance/directory')}
    ))
```

You can force absolute paths with the `absolute_paths` context parameter

```{code-cell}
print_json(
  model.model_dump_json(
    round_trip=True, 
    context={"absolute_paths": True}
    ))
```

#### Durable Interface Metadata

Numpydantic tries to be [stable](./development.md#api-stability),
but we're not perfect. To preserve the full information about the
interface that's needed to load the data referred to by the value,
use the `mark_interface` contest parameter:

```{code-cell}
print_json(
  model.model_dump_json(
    round_trip=True, 
    context={"mark_interface": True}
    ))
```

```{todo}
We will also add a separate `mark_version` parameter for marking
the specific version of the relevant interface package, like `zarr`, or `numpy`,
patience.
```

## Context parameters

A reference listing of all the things that can be passed to
{func}`~pydantic.BaseModel.model_dump_json`


### `mark_interface`

Nest an additional layer of metadata for unambigous serialization that
can be absolutely resolved across numpydantic versions 
(for now for downstream metadata purposes only, 
automatically resolving to a numpydantic version is not yet possible.)

Supported interfaces:

- (all)

```{code-cell}
model = MyModel(array=[[1,2],[3,4]])
data = model.model_dump_json(
  round_trip=True,
  context={"mark_interface": True}
)
print_json(data)
```

### `absolute_paths`

Make all paths (that exist) absolute.

Supported interfaces:

- (all)

```{code-cell}
model = MyModel(array=("docs/data/test.h5", "/data"))
data = model.model_dump_json(
    round_trip=True, 
    context={"absolute_paths": True}
    )
print_json(data)
```

### `relative_to`

Make all paths (that exist) relative to the given path

Supported interfaces:

- (all)

```{code-cell}
model = MyModel(array=("docs/data/test.h5", "/data"))
data = model.model_dump_json(
    round_trip=True, 
    context={"relative_to": Path('../')}
    )
print_json(data)
```

### `dump_array`

Dump the raw array contents when serializing to json inside an `array` field

Supported interfaces:
- {class}`.ZarrInterface`

```{code-cell}
model = MyModel(array=("docs/data/test.zarr",))
data = model.model_dump_json(
    round_trip=True, 
    context={"dump_array": True}
    )
print_json(data)
```



[^normalstyle]: o ya we're posting JSON [normal style](https://normal.style)
[^notenforced]: This is only *functionally* enforced at the moment, where 
  a roundtrip test confirms that dtype and type are preserved,
  but there is no formal test for each interface having its own serialization class



