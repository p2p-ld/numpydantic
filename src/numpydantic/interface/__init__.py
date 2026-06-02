"""
Interfaces between nptyping types and array backends
"""

from numpydantic.interface.dask import DaskInterface
from numpydantic.interface.hdf5 import H5ArrayPath, H5Interface
from numpydantic.interface.interface import (
    Interface,
    InterfaceMark,
    JsonDict,
    MarkedJson,
    Proxy,
)
from numpydantic.interface.numpy import NumpyInterface
from numpydantic.interface.typing import ConstructorSpec, InterfaceTyping
from numpydantic.interface.video import VideoInterface
from numpydantic.interface.zarr import ZarrArrayPath, ZarrInterface

__all__ = [
    "ConstructorSpec",
    "DaskInterface",
    "H5ArrayPath",
    "H5Interface",
    "Interface",
    "InterfaceMark",
    "InterfaceTyping",
    "JsonDict",
    "MarkedJson",
    "NumpyInterface",
    "Proxy",
    "VideoInterface",
    "ZarrArrayPath",
    "ZarrInterface",
]
