# Changelog

## 1.*

### 1.2.3 - 24-07-31 - Vendor `nptyping`

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