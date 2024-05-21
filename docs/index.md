# numpydantic

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
- **Interfaces** - Works with {mod}`~.interface.numpy`, {mod}`~.interface.dask`, {mod}`~.interface.hdf5`, {mod}`~.interface.zarr`, 
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
- (see [todo](./todo.md))

## Usage

Specify an array using [nptyping syntax](https://github.com/ramonhagenaars/nptyping/blob/master/USERDOCS.md)
and use it with your favorite array library :)

Use the {class}`~numpydantic.NDArray` class like you would any other python type,
combine it with {class}`typing.Union`, make it {class}`~typing.Optional`, etc.

For example, to support a 

```python
from typing import Union
from pydantic import BaseModel
import numpy as np

from numpydantic import NDArray, Shape

class Image(BaseModel):
    """
    Images: grayscale, RGB, RGBA, and videos too!
    """
    array: Union[
        NDArray[Shape["* x, * y"], np.uint8],
        NDArray[Shape["* x, * y, 3 rgb"], np.uint8],
        NDArray[Shape["* t, * x, * y, 4 rgba"], np.float64]
    ]
```

And then use that as a transparent interface to your favorite array library!

### Numpy

The Coca-Cola of array libraries

```python
import numpy as np
# works
frame_gray = Image(array=np.ones((1280, 720), dtype=np.uint8))
frame_rgb  = Image(array=np.ones((1280, 720, 3), dtype=np.uint8))
frame_rgba = Image(array=np.ones((1280, 720, 4), dtype=np.uint8))
video_rgb  = Image(array=np.ones((100, 1280, 720, 3), dtype=np.uint8))

# fails
wrong_n_dimensions = Image(array=np.ones((1280,), dtype=np.uint8))
wrong_shape = Image(array=np.ones((1280,720,10), dtype=np.uint8))
wrong_type = Image(array=np.ones((1280,720,3), dtype=np.float64))

# shapes and types are checked together, so..
# this works
float_video = Image(array=np.ones((100, 1280, 720, 4), dtype=float))
# this doesn't
wrong_shape_float_video = Image(array=np.ones((100, 1280, 720, 3), dtype=float))
```

### Dask

High performance chunked arrays! The backend for many new array libraries! 

Works exactly the same as numpy arrays

```python
import dask.array as da

# validate a huge video
video_array = da.zeros(shape=(1920,1080,1000000,3), dtype=np.uint8)

# this works
dask_video = Image(array=video_array)
```

### HDF5

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
array = np.random.random((1920,1080,3)).astype(np.uint8)
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
>>> # do some stuff that requires the datset to be held open
>>> h5f_image.array.close()
```

### Zarr

Zarr works similarly!

Use it with any of Zarr's backends: Nested, Zipfile, S3, it's all the same!

```{todo}
Add the zarr examples!
```




```{toctree}
:maxdepth: 2
:caption: Contents
:hidden: true

overview
ndarray
hooks
todo
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden: true

api/interface/index
api/index
api/dtype
api/ndarray
api/maps
api/monkeypatch
api/types

```

