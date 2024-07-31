# Changelog

## 1.*

### 1.2.2 - 24-07-31

Add `datetime` map to numpy's :class:`numpy.datetime64` type

### 1.2.1 - 24-06-27

Fix a minor bug where {class}`~numpydantic.exceptions.DtypeError` would not cause
pydantic to throw a {class}`pydantic.ValidationError` because custom validator functions
need to raise either `AssertionError` or `ValueError` - made `DtypeError` also
inherit from `ValueError` because that is also technically true.

### 1.2.0 - 24-06-13 - Shape ranges

- Add ability to specify shapes as ranges - see [shape ranges](shape-ranges)

### 1.1.0 - 24-05-24 - Instance Checking

https://github.com/p2p-ld/numpydantic/pull/1

Features:
- Add `__instancecheck__` method to NDArrayMeta to support `isinstance()` validation
- Add finer grained errors and parent classes for validation exceptions
- Add fast matching mode to {meth}`.Interface.match` that returns the first match without checking for duplicate matches

Bugfix:
- get all interface classes recursively, instead of just first-layer children 
- fix stubfile generation which badly handled `typing` imports.