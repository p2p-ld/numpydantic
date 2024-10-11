# Changelog

## 1.*

### 1.6.*

#### 1.6.4 - 24-10-11 - Combinatoric Testing

PR: https://github.com/p2p-ld/numpydantic/pull/31


We have rewritten our testing system for more rigorous tests,
where before we were limited to only testing dtype or shape cases one at a time,
now we can test all possible combinations together!

This allows us to have better guarantees for behavior that all interfaces
should support, validating it against all possible dtypes and shapes.

We also exposed all the helpers and array testing classes for downstream development
so that it would be easier to test and validate any 3rd-party interfaces
that haven't made their way into mainline numpydantic yet - 
see the {mod}`numpydantic.testing` module.

See the [testing documentation](./contributing/testing.md) for more details.

**Bugfix**
- Previously, numpy and dask arrays with a model dtype would fail json roundtripping
  because they wouldn't be correctly cast back to the model type. Now they are.
- Zarr would not dump the dtype of an array when it roundtripped to json,
  causing every array to be interpreted as a random integer or float type.
  `dtype` is now dumped and used when deserializing.


#### 1.6.3 - 24-09-26

**Bugfix**
- h5py v3.12.0 was actually fine, but we did need to change the way that
  the hdf5 tests work to not hold the file open during the test. Easy enough change.
  the version cap has been removed from h5py (which is optional anyway,
  so any version could be installed separately)

#### 1.6.2 - 24-09-25

Very minor bugfix and CI release

PR: https://github.com/p2p-ld/numpydantic/pull/26

**Bugfix**
- h5py v3.12.0 broke file locking, so a temporary maximum version cap was added
  until that is resolved. See [`h5py/h5py#2506`](https://github.com/h5py/h5py/issues/2506)
  and [`#27`](https://github.com/p2p-ld/numpydantic/issues/27)
- The `_relativize_paths` function used in roundtrip dumping was incorrectly
  relativizing paths that are intended to refer to paths within a dataset,
  rather than a file. This, as well as windows-specific bugs was fixed so that
  directories that exist but are just below the filesystem root (like `/data`)
  are excluded. If this becomes a problem then we will have to make the
  relativization system a bit more robust by specifically enumerating which
  path-like things are *not* intended to be paths.

**CI**
- `numpydantic` was added as an array range generator in `linkml` 
  ([`linkml/linkml#2178`](https://github.com/linkml/linkml/pull/2178)),
  so tests were added to ensure that changes to `numpydantic` don't break
  linkml array range generation. `numpydantic`'s tests are naturally a
  superset of the behavior tested in `linkml`, but this is a good
  paranoia check in case we drift substantially (which shouldn't happen).

#### 1.6.1 - 24-09-23 - Support Union Dtypes

It's now possible to do this, like it always should have been

```python
class MyModel(BaseModel):
    array: NDArray[Any, int | float]
```

**Features** 
- Support for Union Dtypes

**Structure**
- New `validation` module containing `shape` and `dtype` convenience methods
  to declutter main namespace and make a grouping for related code
- Rename all serialized arrays within a container dict to `value` to be able
  to identify them by convention and avoid long iteration - see perf below.

**Perf**
- Avoid iterating over every item in an array trying to convert it to a path for
  a several order of magnitude perf improvement over `1.6.0` (oops)

**Docs**
- Page for `dtypes`, mostly stubs at the moment, but more explicit documentation
  about what kind of dtypes we support.


#### 1.6.0 - 24-09-23 - Roundtrip JSON Serialization

Roundtrip JSON serialization is here - with serialization to list of lists,
as well as file references that don't require copying the whole array if 
used in data modeling, control over path relativization, and stamping of
interface version for the extra provenance conscious.

Please see [serialization](./serialization.md) for narrative documentation :)

**Potentially Breaking Changes**
- See [development](./development.md) for a statement about API stability
- An additional {meth}`.Interface.deserialize` method has been added to
  {meth}`.Interface.validate` - downstream users are not intended to override the
  `validate method`, but if they have, then JSON deserialization will not work for them.
- `Interface` subclasses now require a `name` attribute, a short string identifier for that interface,
  and a `json_model` that inherits from {class}`.interface.JsonDict`. Interfaces without
  these attributes will not be able to be instantiated.
- {meth}`.Interface.to_json` is now an abstract method that all interfaces must define.

**Features**
- Roundtrip JSON serialization - by default dump to a list of list arrays, but
  support the `round_trip` keyword in `model_dump_json` for provenance-preserving dumps
- JSON Schema generation has been separated from `core_schema` generation in {class}`.NDArray`.
  Downstream interfaces can customize json schema generation without compromising ability to validate.
- All proxy classes must have an `__eq__` dunder method to compare equality -
  in proxy classes, these compare equality of arguments, since the arrays that
  are referenced on disk should be equal by definition. Direct array comparison
  should use {func}`numpy.array_equal`
- Interfaces previously couldn't be instantiated without explicit shape and dtype arguments,
  these have been given `Any` defaults.
- New {mod}`numpydantic.serialization` module to contain serialization logic.

**New Classes**
See the docstrings for descriptions of each class
- `MarkMismatchError` for when an array serialized with `mark_interface` doesn't match
  the interface that's deserializing it
- {class}`.interface.InterfaceMark`
- {class}`.interface.MarkedJson`
- {class}`.interface.JsonDict`
  - {class}`.dask.DaskJsonDict`
  - {class}`.hdf5.H5JsonDict`
  - {class}`.numpy.NumpyJsonDict`
  - {class}`.video.VideoJsonDict`
  - {class}`.zarr.ZarrJsonDict`

**Bugfix**
- [`#17`](https://github.com/p2p-ld/numpydantic/issues/17) - Arrays are re-validated as lists, rather than arrays
- Some proxy classes would fail to be serialized becauase they lacked an `__array__` method.
  `__array__` methods have been added, and tests for coercing to an array to prevent regression.
- Some proxy classes lacked a `__name__` attribute, which caused failures to serialize
  when the `__getattr__` methods attempted to pass it through. These have been added where needed.

**Docs**
- Add statement about versioning and API stability to [development](./development.md)
- Add docs for serialization!
- Remove stranded docs from hooks and monkeypatch
- Added `myst_nb` to docs dependencies for direct rendering of code and output

**Tests**
- Marks have been added for running subsets of the tests for a given interface,
  package feature, etc.
- Tests for all the above functionality



### 1.5.*

#### 1.5.3 - 24-09-03 - Bugfix, type checking for empty HDF5 datasets

- [#16](https://github.com/p2p-ld/numpydantic/pull/16): Empty HDF5 datasets shouldn't break validation
  if the NDArray spec allows Any shaped arrays.

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