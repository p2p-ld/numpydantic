# Development

## Versioning

This package uses a colloquial form of [semantic versioning 2](https://semver.org/).

Specifically:

- Major version `2.*.*` is reserved for the transition from nptyping to using
  `TypeVarTuple`, `Generic`, and `Protocol`. Until `2.*.*`...
  - breaking changes will be indicated with an advance in `MINOR`
      version, taking the place of `MAJOR` in semver
  - backwards-compatible bugfixes **and** additions in functionality
    will be indicated by a `PATCH` release, taking the place of `MINOR` and
    `PATCH` in semver.
- After `2.*.*`, semver as usual will resume

You are encouraged to set an upper bound on your version dependencies until
we pass `2.*.*`, as the major function of numpydantic is stable,
but there is still a decent amount of jostling things around to be expected.


### API Stability

- All breaking changes to the **public API** will be signaled by a major
  version's worth of deprecation warnings
- All breaking changes to the **development API** will be signaled by a
  minor version's worth of deprecation warnings.
- Changes to the remainder of the package, whether marked as private with a
  leading underscore or not, including the import structure of the package,
  are not considered part of the API and should not be relied on as stable 
  until explicitly marked otherwise.

#### Public API

**Only the {class}`.NDArray` and {class}`.Shape` classes should be considered
part of the stable public API.**

All associated functionality for validation should also be considered
a stable part of the `NDArray` and `Shape` classes - functionality
will only be added here, and the departure for the string-form of the 
shape specifications (and its removal) will take place in `v3.*.*`

End-users of numpydantic should pin an upper bound for the `MAJOR` version
until after `v2.*.*`, after which time it is up to your discretion - 
no breaking changes are planned, but they would be signaled by a major version change.

#### Development API

**Only the {class}`.Interface` class and its subclasses, 
along with the Public API,
should be considered part of the stable development API.**

The `Interface` class is the primary point of external development expected
for numpydantic. It is still somewhat in flux, but it is prioritized for stability
and deprecation warnings above the rest of the package. 

Dependent packages that define their own `Interface`s should pin an upper
bound for the `PATCH` version until `2.*.*`, and afterwards likely pin a `MINOR` version.
Tests are designed such that it should be easy to test major features against
each interface, and that work is also ongoing. Once the test suite reaches
maturity, it should be possible for any downstream interfaces to simply use those to
ensure they are compatible with the latest version.

## Release Schedule

There is no release schedule. Versions are released according to need and available labor.

## Contributing

### Dev environment

```{todo}
Document dev environment

Really it's very simple, you just clone a fork and install
the `dev` environment like `pip install '.[dev]'`
```

### Pull Requests

```{todo}
Document pull requests if we ever receive one
```
