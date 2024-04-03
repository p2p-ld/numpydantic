from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from nptyping import NDArray as _NDArray
from pydantic_core import core_schema


class NDArrayProxy:
    """
    Thin proxy to numpy arrays stored within hdf5 files,
    only read into memory when accessed, but otherwise
    passthrough all attempts to access attributes.
    """

    def __init__(self, h5f_file: Path | str, path: str):
        """
        Args:
            h5f_file (:class:`pathlib.Path`): Path to source HDF5 file
            path (str): Location within HDF5 file where this array is located
        """
        self.h5f_file = Path(h5f_file)
        self.path = path

    def __getattr__(self, item) -> Any:
        with h5py.File(self.h5f_file, "r") as h5f:
            obj = h5f.get(self.path)
            return getattr(obj, item)

    def __getitem__(self, slice: slice) -> np.ndarray:
        with h5py.File(self.h5f_file, "r") as h5f:
            obj = h5f.get(self.path)
            return obj[slice]

    def __setitem__(self, slice, value) -> None:
        raise NotImplementedError("Cant write into an arrayproxy yet!")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: _NDArray,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        from numpydantic import NDArray

        return NDArray.__get_pydantic_core_schema__(cls, _source_type, _handler)
