# numpydantic

[![PyPI - Version](https://img.shields.io/pypi/v/numpydantic)](https://pypi.org/project/numpydantic)
[![Documentation Status](https://readthedocs.org/projects/numpydantic/badge/?version=latest)](https://numpydantic.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/p2p-ld/numpydantic/badge.svg)](https://coveralls.io/github/p2p-ld/numpydantic)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python package for specifying, validating, and serializing arrays with arbitrary backends in pydantic.

**Problem:** 
1) Pydantic is great for modeling data. 
2) Arrays are one of a few elemental types in computing,

but ...

3) if you try and specify an array in pydantic, this happens:

```python
>>> from pydantic import BaseModel
>>> import numpy as np

>>> class MyModel(BaseModel):
>>>     array: np.ndarray
pydantic.errors.PydanticSchemaGenerationError: 
Unable to generate pydantic-core schema for <class 'numpy.ndarray'>. 
Set `arbitrary_types_allowed=True` in the model_config to ignore this error 
or implement `__get_pydantic_core_schema__` on your type to fully support it.
```

And setting `arbitrary_types_allowed = True` still prohibits you from 
generating JSON Schema, serialization to JSON

## Features:
- **Types** - Annotations (based on [npytyping](https://github.com/ramonhagenaars/nptyping))
  for specifying arrays in pydantic models
- **Validation** - Shape, dtype, and other array validations
- **Interfaces** - Works with [`numpy`](https://numpydantic.readthedocs.io/en/latest/api/interface/numpy.html), 
  [`dask`](https://numpydantic.readthedocs.io/en/latest/api/interface/dask.html), 
  [`hdf5`](https://numpydantic.readthedocs.io/en/latest/api/interface/hdf5.html),
  [`video`](https://numpydantic.readthedocs.io/en/latest/api/interface/video.html), 
  [`zarr`](https://numpydantic.readthedocs.io/en/latest/api/interface/zarr.html),
  and a simple extension system to make it work with whatever else you want!
- **Serialization** - Dump an array as a JSON-compatible array-of-arrays with enough metadata to be able to 
  recreate the model in the native format
- **Schema Generation** - Correct JSON Schema for arrays, complete with shape and dtype constraints, to
  make your models interoperable 

Coming soon:
- **Metadata** - This package was built to be used with [linkml arrays](https://linkml.io/linkml/schemas/arrays.html),
  so we will be extending it to include arbitrary metadata included in the type annotation object in the JSON schema representation.
- **Extensible Specification** - for v1, we are implementing the existing nptyping syntax, but 
  for v2 we will be updating that to an extensible specification syntax to allow interfaces to validate additional
  constraints like chunk sizes, as well as make array specifications more introspectable and friendly to runtime usage.
- **Advanced dtype handling** - handling dtypes that only exist in some array backends, allowing
  minimum and maximum precision ranges, and so on as type maps provided by interface classes :)
- (see [todo](https://numpydantic.readthedocs.io/en/latest/todo.html))

## Installation

numpydantic tries to keep dependencies minimal, so by default it only comes with 
dependencies to use the numpy interface. Add the extra relevant to your favorite
array library to be able to use it!

```shell
pip install numpydantic
# dask
pip install 'numpydantic[dask]'
# hdf5
pip install 'numpydantic[hdf5]'
# video
pip install 'numpydantic[video]'
# zarr
pip install 'numpydantic[zarr]'
# all array formats
pip intsall 'numpydantic[array]'
```

## Usage

Specify an array using [nptyping syntax](https://github.com/ramonhagenaars/nptyping/blob/master/USERDOCS.md)
and use it with your favorite array library :)

Use the `NDArray` class like you would any other python type,
combine it with `Union`, make it `Optional`, etc.

For example, to specify a very special type of image that can either be
- a 2D float array where the axes can be any size, or 
- a 3D uint8 array where the third axis must be size 3
- a 1080p video 

```python
from typing import Union
from pydantic import BaseModel
import numpy as np

from numpydantic import NDArray, Shape

class Image(BaseModel):
    array: Union[
        NDArray[Shape["* x, * y"], float],
        NDArray[Shape["* x, * y, 3 rgb"], np.uint8],
        NDArray[Shape["* t, 1080 y, 1920 x, 3 rgb"], np.uint8]
    ]
```

And then use that as a transparent interface to your favorite array library!

### Interfaces

#### Numpy

The Coca-Cola of array libraries

```python
import numpy as np
# works
frame_gray = Image(array=np.ones((1280, 720), dtype=float))
frame_rgb  = Image(array=np.ones((1280, 720, 3), dtype=np.uint8))

# fails
wrong_n_dimensions = Image(array=np.ones((1280,), dtype=float))
wrong_shape = Image(array=np.ones((1280,720,10), dtype=np.uint8))

# shapes and types are checked together, so this also fails
wrong_shape_dtype_combo = Image(array=np.ones((1280, 720, 3), dtype=float))
```

#### Dask

High performance chunked arrays! The backend for many new array libraries! 

Works exactly the same as numpy arrays

```python
import dask.array as da

# validate a humongous image without having to load it into memory
video_array = da.zeros(shape=(1e10,1e20,3), dtype=np.uint8)
dask_video = Image(array=video_array)
```

#### HDF5

Array work increasingly can't fit on memory, but dealing with arrays on disk 
can become a pain in concurrent applications. Numpydantic allows you to 
specify the location of an array within an hdf5 file on disk and use it just like
any other array!

eg. Make an array on disk...

```python
from pathlib import Path
import h5py
from numpydantic.interface.hdf5 import H5ArrayPath

h5f_file = Path('my_file.h5')
array_path = "/nested/array"

# make an HDF5 array
h5f = h5py.File(h5f_file, "w")
array = np.random.randint(0, 255, (1920,1080,3), np.uint8)
h5f.create_dataset(array_path, data=array)
h5f.close()
```

Then use it in your model! numpydantic will only open the file as long as it's needed

```python
>>> h5f_image = Image(array=H5ArrayPath(file=h5f_file, path=array_path))
>>> h5f_image.array[0:5,0:5,0]
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]], dtype=uint8)
>>> h5f_image.array[0:2,0:2,0] = 1
>>> h5f_image.array[0:5,0:5,0]
array([[1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]], dtype=uint8)
```

Numpydantic tries to be a smart but transparent proxy, exposing the methods and attributes
of the source type even when we aren't directly using them, like when dealing with on-disk HDF5 arrays.

If you want, you can take full control and directly interact with the underlying :class:`h5py.Dataset`
object and leave the file open between calls:

```python
>>> dataset = h5f_image.array.open()
>>> # do some stuff that requires the dataset to be held open
>>> h5f_image.array.close()
```

#### Video

Videos are just arrays with fancy encoding! Numpydantic can validate shape and dtype
as well as lazy load chunks of frames with arraylike syntax!

Say we have some video `data.mp4` ...

```python
video = Image(array='data.mp4')
# get a single frame
video.array[5]
# or a range of frames!
video.array[5:10]
# or whatever slicing you want to do!
video.array[5:50:5, 0:10, 50:70]
```

As elsewhere, a proxy class is a transparent pass-through interface to the underlying
opencv class, so we can get the rest of the video properties ...

```python
import cv2

# get the total frames from opencv
video.array.get(cv2.CAP_PROP_FRAME_COUNT)
# the proxy class also provides a convenience property
video.array.n_frames
```

#### Zarr

Zarr works similarly!

Use it with any of Zarr's backends: Nested, Zipfile, S3, it's all the same!

Eg. create a nested zarr array on disk and use it...

```python
import zarr
from numpydantic.interface.zarr import ZarrArrayPath

array_file = 'data/array.zarr'
nested_path = 'data/sets/here'

root = zarr.open(array_file, mode='w')
nested_array = root.zeros(
    nested_path, 
    shape=(1000, 1080, 1920, 3), 
    dtype=np.uint8
)

# validates just fine!
zarr_video = Image(array=ZarrArrayPath(array_file, nested_path))
# or just pass a tuple, the interface can discover it's a zarr array
zarr_video = Image(array=(array_file, nested_path))
```

### JSON Schema

Numpydantic generates JSON Schema for all its array specifications, so for the above
model, we get a schema for each of the possible array types that properly handles
the shape and dtype constraints and includes the origin numpy type as a `dtype` annotation.

```python
Image.model_json_schema()
```

```json
{
  "properties": {
    "array": {
      "anyOf": [
        {
          "items": {"items": {"type": "number"}, "type": "array"},
          "type": "array"
        },
        {
          "dtype": "numpy.uint8",
          "items": {
            "items": {
              "items": {
                "maximum": 255,
                "minimum": 0,
                "type": "integer"
              },
              "maxItems": 3,
              "minItems": 3,
              "type": "array"
            },
            "type": "array"
          },
          "type": "array"
        },
        {
          "dtype": "numpy.uint8",
          "items": {
            "items": {
              "items": {
                "items": {
                  "maximum": 255,
                  "minimum": 0,
                  "type": "integer"
                },
                "maxItems": 3,
                "minItems": 3,
                "type": "array"
              },
              "maxItems": 1920,
              "minItems": 1920,
              "type": "array"
            },
            "maxItems": 1080,
            "minItems": 1080,
            "type": "array"
          },
          "type": "array"
        }
      ],
      "title": "Array"
    }
  },
  "required": ["array"],
  "title": "Image",
  "type": "object"
}
```

numpydantic can even handle shapes with unbounded numbers of dimensions by using
recursive JSON schema!!!

So the any-shaped array (using nptyping's ellipsis notation):

```python
class AnyShape(BaseModel):
    array: NDArray[Shape["*, ..."], np.uint8]
```

is rendered to JSON-Schema like this:

```json
{
  "$defs": {
    "any-shape-array-9b5d89838a990d79": {
      "anyOf": [
        {
          "items": {
            "$ref": "#/$defs/any-shape-array-9b5d89838a990d79"
          },
          "type": "array"
        },
        {"maximum": 255, "minimum": 0, "type": "integer"}
      ]
    }
  },
  "properties": {
    "array": {
      "dtype": "numpy.uint8",
      "items": {"$ref": "#/$defs/any-shape-array-9b5d89838a990d79"},
      "title": "Array",
      "type": "array"
    }
  },
  "required": ["array"],
  "title": "AnyShape",
  "type": "object"
}
```

where the key `"any-shape-array-9b5d89838a990d79"` uses a (blake2b) hash of the
inner dtype specification so that having multiple any-shaped arrays in a single 
model schema are deduplicated without conflicts.

### Dumping

One of the main reasons to use chunked array libraries like zarr is to avoid
needing to load the entire array into memory. When dumping data to JSON, numpydantic 
tries to mirror this behavior, by default only dumping the metadata that is
necessary to identify the array.

For example, with zarr:

```python
array = zarr.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
instance = Image(array=array)
dumped = instance.model_dump_json()
```

```json
{
  "array":
  {
    "Chunk shape": "(3, 3)",
    "Chunks initialized": "1/1",
    "Compressor": "Blosc(cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0)",
    "Data type": "float64",
    "No. bytes": "72",
    "No. bytes stored": "421",
    "Order": "C",
    "Read-only": "False",
    "Shape": "(3, 3)",
    "Storage ratio": "0.2",
    "Store type": "zarr.storage.KVStore",
    "Type": "zarr.core.Array",
    "hexdigest": "c51604eace325fe42bbebf39146c0956bd2ed13c"
  }
}
```

To print the whole array, we use pydantic's serialization contexts:

```python
dumped = instance.model_dump_json(context={'zarr_dump_array': True})
```
```json
{
  "array":
  {
    "same thing,": "except also...",
    "array": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    "hexdigest": "c51604eace325fe42bbebf39146c0956bd2ed13c"
  }
}
```