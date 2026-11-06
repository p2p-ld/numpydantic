# Writing an Interface

An interface lets numpydantic validate and serialize a new kind of array.
See [Interfaces](../interfaces.md) for how interfaces are matched and how validation works.
The existing interfaces in `src/numpydantic/interface` are good examples to start from.

## Interface Class

Subclass {class}`.Interface` and set these class attributes:

- {attr}`~.Interface.name` - a short name for the interface
- {attr}`~.Interface.input_types` - the types the interface accepts as input
- {attr}`~.Interface.return_type` - the type the interface returns after validation
- {attr}`~.Interface.json_model` - a {class}`.JsonDict` subclass used to
  roundtrip the array through JSON (see [Serialization](../serialization.md))

and implement these methods:

- {meth}`~.Interface.check` - whether a given input should use this interface
- {meth}`~.Interface.enabled` - whether the interface can be used,
  usually whether its dependencies are installed
- {meth}`~.Interface.to_json` - convert the array to JSON-compatible python types

Then override any of the [validation methods](../interfaces.md#validation)
the array format needs.

Import the interface in `src/numpydantic/interface/__init__.py`
and add it to `__all__`.

## Dependencies

The array library should be an optional dependency.
Add an extra for it under `[project.optional-dependencies]` in `pyproject.toml`
and add that extra to the `arrays` extra.
Since the library might not be installed, {meth}`~.Interface.enabled`
should check for it rather than the module importing it unconditionally.

## Tests

1. Add an {class}`.InterfaceCase` for the interface in `src/numpydantic/testing/interfaces.py`.
   If the array format has several forms (e.g. zarr arrays in memory, in a directory,
   or in a zip file), add a case for each.
2. Add a {class}`.ValidationCase` for each of those cases to `INTERFACE_CASES`
   in `src/numpydantic/testing/cases.py`.
3. Add a pytest marker with the interface's name to `[tool.pytest.ini_options]`
   in `pyproject.toml`.
4. Add tests for anything specific to the interface in `tests/test_interface`.

See [Testing](./testing.md) for more on interface and validation cases.

## Static Typing

Optionally, an interface can set {attr}`~.Interface.typing` to an
{class}`~.interface.typing.InterfaceTyping` subclass so that the mypy plugin
can infer shapes and dtypes from the library's array constructors.
See [Constructor Inference](../typecheckers.md#constructor-inference).
