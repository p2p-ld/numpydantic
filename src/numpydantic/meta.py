"""
Metaprogramming functions for numpydantic to modify itself :)
"""

from pathlib import Path
from warnings import warn

from numpydantic.interface import Interface

_BUILTIN_IMPORTS = ("import typing", "import pathlib")


def generate_ndarray_stub() -> str:
    """
    Make a stub file based on the array interfaces that are available
    """

    import_strings = []
    type_names = []
    for arr in Interface.input_types():
        if arr.__module__ == "builtins":
            continue

        # Create import statements, saving aliased name of type if needed
        if arr.__module__.startswith("numpydantic") or arr.__module__ == "typing":
            type_name = str(arr) if arr.__module__ == "typing" else arr.__name__
            if arr.__module__ != "typing":
                import_strings.append(f"from {arr.__module__} import {type_name}")
        else:
            # since other packages could use the same name for an imported object
            # (eg dask and zarr both use an Array class)
            # we make an import alias from the module names to differentiate them
            # in the type annotation
            mod_name = "".join([a.capitalize() for a in arr.__module__.split(".")])
            type_name = mod_name + arr.__name__
            import_strings.append(
                f"from {arr.__module__} import {arr.__name__} " f"as {type_name}"
            )

        type_names.append(type_name)

    import_strings.extend(_BUILTIN_IMPORTS)
    import_strings = list(dict.fromkeys(import_strings))
    import_string = "\n".join(import_strings)

    class_union = " | ".join(type_names)
    ndarray_type = "NDArray = " + class_union

    stub_string = "\n".join([import_string, ndarray_type])
    return stub_string


def update_ndarray_stub() -> None:
    """
    Update the ndarray.pyi string in the numpydantic file
    """
    from numpydantic import ndarray

    try:
        stub_string = generate_ndarray_stub()

        pyi_file = Path(ndarray.__file__).with_suffix(".pyi")
        with open(pyi_file, "w") as pyi:
            pyi.write(stub_string)
    except Exception as e:  # pragma: no cover
        warn(f"ndarray.pyi stub file could not be generated: {e}", stacklevel=1)
