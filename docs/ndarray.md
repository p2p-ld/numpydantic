# Constrained Arrays

## Implementation details

```{todo}
**Docs:**

Describe implementation details!
```

## Examples

### Declaration

Type with a single {class}`~numpydantic.NDArray` class, or use a {class}`~typing.Union`
to express more complex array constraints.

This package is effectively a Pydantic interface to [nptyping](https://github.com/ramonhagenaars/nptyping),
so any array syntax is valid there. (see [TODO](todo) for caveats)

```python
from typing import Union
from pydantic import BaseModel
from src.numpydantic import NDArray, Shape, UInt8, Float, Int


class Image(BaseModel):
    """
    Data values. Data can be in 1-D, 2-D, 3-D, or 4-D. The first dimension should always represent time. This can also be used to store binary data (e.g., image frames). This can also be a link to data stored in an external file.
    """
    array: Union[
        NDArray[Shape["* x, * y"], UInt8],
        NDArray[Shape["* x, * y, 3 rgb"], UInt8],
        NDArray[Shape["* x, * y, 4 rgba"], UInt8],
        NDArray[Shape["* t, * x, * y, 3 rgb"], UInt8],
        NDArray[Shape["* t, * x, * y, 4 rgba"], Float]
    ]
```

### Validation:

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

# shapes and types are checked together
float_video = Image(array=np.ones((100, 1280, 720, 4),dtype=float))
wrong_shape_float_video = Image(array=np.ones((100, 1280, 720, 3),dtype=float))
```

### JSON schema generation:

```python
class MyArray(BaseModel):
  array: NDArray[Shape["2 x, * y, 4 z"], Float]
```

```python
>>> print(json.dumps(MyArray.model_json_schema(), indent=2))
```

```json
{
  "properties": {
    "array": {
      "items": {
        "items": {
          "items": {
            "type": "number"
          },
          "maxItems": 4,
          "minItems": 4,
          "type": "array"
        },
        "type": "array"
      },
      "maxItems": 2,
      "minItems": 2,
      "title": "Array",
      "type": "array"
    }
  },
  "required": [
    "array"
  ],
  "title": "MyArray",
  "type": "object"
}
```

### Serialization

```python
class SmolArray(BaseModel):
    array: NDArray[Shape["2 x, 2 y"], Int]

class BigArray(BaseModel):
    array: NDArray[Shape["1000 x, 1000 y"], Int]
```

Serialize small arrays as lists of lists, and big arrays as a b64-encoded blosc compressed string

```python
>>> smol = SmolArray(array=np.array([[1,2],[3,4]], dtype=int))
>>> big = BigArray(array=np.random.randint(0,255,(1000,1000),int))

>>> print(smol.model_dump_json())
{"array":[[1,2],[3,4]]}
>>> print(big.model_dump_json())
{
  "array": "( long b64 encoded string )",
  "shape": [1000, 1000],
  "dtype": "int64",
  "unpack_fns": ["base64.b64decode", "blosc2.unpack_array2"],
}
```

## TODO

```{todo}
Implement structured arrays
```

```{todo}
Implement pandas dataframe validation?
```