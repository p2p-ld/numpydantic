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

    import_strings = [
        f"from {arr.__module__} import {arr.__name__}"
        for arr in Interface.input_types()
        if arr.__module__ != "builtins"
    ]
    import_strings.extend(_BUILTIN_IMPORTS)
    import_string = "\n".join(import_strings)

    class_names = [
        arr.__name__ if arr.__module__ != "typing" else str(arr)
        for arr in Interface.input_types()
    ]
    class_union = " | ".join(class_names)
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
    except Exception as e:
        warn(f"ndarray.pyi stub file could not be generated: {e}", stacklevel=1)
