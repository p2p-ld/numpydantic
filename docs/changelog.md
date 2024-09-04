# Changelog

## 1.*

### 1.5.*

#### 1.5.2 - 24-09-03 - `datetime` support for HDF5

- [#15](https://github.com/p2p-ld/numpydantic/pull/15): Datetimes are supported as 
  dtype annotations for HDF5 arrays when encoded as `S32` isoformatted byte strings

#### 1.5.1 - 24-09-03 - Fix revalidation with proxy classes

Bugfix:
- [#14](https://github.com/p2p-ld/numpydantic/pull/14): Allow revalidation of proxied arrays

Tests:
- Add test module for tests against all interfaces, test for above bug

#### 1.5.0 - 24-09-02 - `str` support for HDF5

Strings in hdf5 are tricky! HDF5 doesn't have native support for unicode, 
but it can be persuaded to store data in ASCII or virtualized utf-8 under somewhat obscure conditions.

This PR uses h5py's string methods to expose string datasets (compound or not) 
via the h5proxy with the `asstr()` view method. 
This also allows us to set strings with normal python strings,
although hdf5 datasets can only be created with `bytes` or other non-unicode encodings.

Since numpydantic isn't necessarily a tool for *creating* hdf5 files 
(nobody should be doing that), but rather an interface to them, 
tests are included for reading and validating (unskip the existing string tests) 
as well as setting/getting.

```python
import h5py
import numpy as np
from pydantic import BaseModel
from numpydantic import NDArray
from typing import Any

class MyModel(BaseModel):
  array: NDArray[Any, str]

h5f = h5py.File('my_data.h5', 'w')
data = np.random.random((10,10)).astype(bytes)
_ = h5f.create_dataset('/dataset', data=data)

instance = MyModel(array=('my_data.h5', '/dataset'))
instance[0,0] = 'hey'
assert instance[0,0] == 'hey'
```

### 1.4.*

#### 1.4.1 - 24-09-02 - `len()` support and dunder method testing

It's pretty natural to want to do `len(array)` as a shorthand for `array.shape[0]`, 
but since some of the numpydantic classes are passthrough proxy objects, 
they don't implement all the dunder methods of the classes they wrap 
(though they should attempt to via `__getattr__`). 

This PR adds `__len__` to the two interfaces that are missing it, 
and adds fixtures and makes a testing module specifically for testing dunder methods 
that should be true across all interfaces. 
Previously we have had fixtures that test all of a set of dtype and shape cases for each interface,
but we haven't had a way of asserting that something should be true for all interfaces. 
There is a certain combinatoric explosion when we start testing across all interfaces, 
for all input types, for all dtype and all shape cases, 
but for now numpydantic is fast enough that this doesn't matter <3.

#### 1.4.0 - 24-09-02 - HDF5 Compound Dtype Support

HDF5 can have compound dtypes like:

```python
import numpy as np
import h5py

dtype = np.dtype([("data", "i8"), ("extra", "f8")])
data = np.zeros((10, 20), dtype=dtype)
with h5py.File('mydata.h5', "w") as h5f:
    dset = h5f.create_dataset("/dataset", data=data)

```

```python
>>> dset[0:1]
array([[(0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.),
        (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.),
        (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.), (0, 0.)]],
      dtype=[('data', '<i8'), ('extra', '<f8')])
```

Sometimes we want to split those out to separate fields like this:

```python
class MyModel(BaseModel):
    data: NDArray[Any, np.int64]
    extra: NDArray[Any, np.float64]
```

So that's what 1.4.0 allows, using an additional field in the H5ArrayPath:

```python
from numpydantic.interfaces.hdf5 import H5ArrayPath

my_model = MyModel(
    data = H5ArrayPath(file='mydata.h5', path="/dataset", field="data"),
    extra = H5ArrayPath(file='mydata.h5', path="/dataset", field="extra"),
)

# or just with tuples
my_model = MyModel(
    data = ('mydata.h5', "/dataset", "data"),
    extra = ('mydata.h5', "/dataset", "extra"),
)
```

```python
>>> my_model.data[0,0]
0
>>> my_model.data.dtype
np.dtype('int64')
```

### 1.3.*

#### 1.3.3 - 24-08-13 - Callable type annotations

Problem, when you use a numpydantic `"wrap"` validator, it gives the annotation as a `handler` function. 

So this is effectively what happens

```python
@field_validator("*", mode="wrap")
@classmethod
def cast_specified_columns(
    cls, val: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
) -> Any:
    # where handler is the callable here
    # so 
    # return handler(val)
    
    return NDArray[Any, Any](val)
```

where `Any, Any` is whatever you had put in there. 

So this makes it so you can use an annotation as a functional validator. it looks a little bit whacky but idk it makes sense as a PARAMETERIZED TYPE

```python
>>> from numpydantic import NDArray, Shape
>>> import numpy as np

>>> array = np.array([1,2,3], dtype=int)
>>> validated = NDArray[Shape["3"], int](array)
>>> assert validated is array
True

>>> bad_array = np.array([1,2,3,4], dtype=int)
>>> _ = NDArray[Shape["3"], int](bad_array)
    175 """
    176 Raise a ShapeError if the shape is invalid.
    177 
    178 Raises:
    179     :class:`~numpydantic.exceptions.ShapeError`
    180 """
    181 if not valid:
--> 182     raise ShapeError(
    183         f"Invalid shape! expected shape {self.shape.prepared_args}, "
    184         f"got shape {shape}"
    185     )

ShapeError: Invalid shape! expected shape ['3'], got shape (4,)

```

**Performance:**
- Don't import the pandas module if we don't have to, since we are not
  using it. This shaves ~600ms off import time.


#### 1.3.2 - 24-08-12 - Allow subclasses of dtypes

(also when using objects for dtypes, subclasses of that object are allowed to validate)

#### 1.3.1 - 24-08-12 - Allow arbitrary dtypes, pydantic models as dtypes

Previously we would only allow dtypes if we knew for sure that there was some
python base type to generate a schema with. 

That seems overly restrictive, so relax the requirements to allow
any type to be a dtype. If there are problems with serialization (we assume there will)
or handling the object in a given array framework, we leave that up to the person
who declared the model to handle :). Let people break things and have fun!

Also support the ability to use a pydantic model as the inner type, which works
as expected because pydantic already knows how to generate a schema from its own models.

Only one substantial change, and that is a `get_object_dtype` method which 
interfaces can override if there is some fancy way they have of getting 
types/items from an object array.

#### 1.3.0 - 24-08-05 - Better string dtype handling

API Changes:
- Split apart the validation methods into smaller chunks to better support
  overrides by interfaces. Customize getting and raising errors for dtype and shape,
  as well as separation of concerns between getting, validating, and raising.

Bugfix:
- [#4](https://github.com/p2p-ld/numpydantic/issues/4) - Support dtype checking
  for strings in zarr and numpy arrays

### 1.2.*

#### 1.2.3 - 24-07-31 - Vendor `nptyping`

`nptyping` vendored into `numpydantic.vendor.nptyping` - 
`nptyping` is no longer maintained, and pins `numpy<2`.
It also has many obnoxious warnings and we have to monkeypatch it
so it performs halfway decently. Since we are en-route to deprecating
usage of `nptyping` anyway, in the meantime we have just vendored it in
(it is MIT licensed, included) so that we can make those changes ourselves
and have to patch less of it. Currently the whole package is vendored with 
modifications, but will be whittled away until we have replaced it with
updated type specification system :)

Bugfix:
- [#2](https://github.com/p2p-ld/numpydantic/issues/2) - Support `numpy>=2`
- Remove deprecated numpy dtypes

CI:
- Add windows and mac tests
- Add testing with numpy>=2 and <2

DevOps:
- Make a tox file for local testing, not used in CI.

Tidying:
- Remove `monkeypatch` module! we don't need it anymore!
  everything has either been upstreamed or vendored.

#### 1.2.2 - 24-07-31

Add `datetime` map to numpy's :class:`numpy.datetime64` type

#### 1.2.1 - 24-06-27

Fix a minor bug where {class}`~numpydantic.exceptions.DtypeError` would not cause
pydantic to throw a {class}`pydantic.ValidationError` because custom validator functions
need to raise either `AssertionError` or `ValueError` - made `DtypeError` also
inherit from `ValueError` because that is also technically true.

#### 1.2.0 - 24-06-13 - Shape ranges

- Add ability to specify shapes as ranges - see [shape ranges](shape-ranges)

### 1.1.*

#### 1.1.0 - 24-05-24 - Instance Checking

https://github.com/p2p-ld/numpydantic/pull/1

Features:
- Add `__instancecheck__` method to NDArrayMeta to support `isinstance()` validation
- Add finer grained errors and parent classes for validation exceptions
- Add fast matching mode to {meth}`.Interface.match` that returns the first match without checking for duplicate matches

Bugfix:
- get all interface classes recursively, instead of just first-layer children 
- fix stubfile generation which badly handled `typing` imports.