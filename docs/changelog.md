# Changelog

# 1.*

## 1.1.0 - 24-05-24 - Instance Checking

https://github.com/p2p-ld/numpydantic/pull/1

Features:
- Add `__instancecheck__` method to NDArrayMeta to support `isinstance()` validation
- Add finer grained errors and parent classes for validation exceptions
- Add fast matching mode to {meth}`.Interface.match` that returns the first match without checking for duplicate matches

Bugfix:
- get all interface classes recursively, instead of just first-layer children 
- fix stubfile generation which badly handled `typing` imports.