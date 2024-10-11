from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import dask.array as da
import h5py
import numpy as np
import zarr
from pydantic import BaseModel

from numpydantic.interface import (
    DaskInterface,
    H5ArrayPath,
    H5Interface,
    NumpyInterface,
    VideoInterface,
    ZarrArrayPath,
    ZarrInterface,
)
from numpydantic.testing.helpers import InterfaceCase
from numpydantic.types import DtypeType, NDArrayType


class NumpyCase(InterfaceCase):
    """In-memory numpy array"""

    interface = NumpyInterface

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> np.ndarray:
        if array is not None:
            return np.array(array, dtype=dtype)
        elif issubclass(dtype, BaseModel):
            return np.full(shape=shape, fill_value=dtype(x=1))
        else:
            return np.zeros(shape=shape, dtype=dtype)


class _HDF5MetaCase(InterfaceCase):
    """Base case for hdf5 cases"""

    interface = H5Interface

    @classmethod
    def skip(cls, shape: Tuple[int, ...], dtype: DtypeType) -> bool:
        return issubclass(dtype, BaseModel)


class HDF5Case(_HDF5MetaCase):
    """HDF5 Array"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[H5ArrayPath]:
        if cls.skip(shape, dtype):  # pragma: no cover
            return None

        hdf5_file = path / "h5f.h5"
        array_path = "/" + "_".join([str(s) for s in shape]) + "__" + dtype.__name__
        generator = np.random.default_rng()

        if array is not None:
            data = np.array(array, dtype=dtype)
        elif dtype is str:
            data = generator.random(shape).astype(bytes)
        elif dtype is datetime:
            data = np.empty(shape, dtype="S32")
            data.fill(datetime.now(timezone.utc).isoformat().encode("utf-8"))
        else:
            data = generator.random(shape).astype(dtype)

        h5path = H5ArrayPath(hdf5_file, array_path)

        with h5py.File(hdf5_file, "w") as h5f:
            _ = h5f.create_dataset(array_path, data=data)
        return h5path


class HDF5CompoundCase(_HDF5MetaCase):
    """HDF5 Array with a fake compound dtype"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[H5ArrayPath]:
        if cls.skip(shape, dtype):  # pragma: no cover
            return None

        hdf5_file = path / "h5f.h5"
        array_path = "/" + "_".join([str(s) for s in shape]) + "__" + dtype.__name__
        if array is not None:
            data = np.array(array, dtype=dtype)
        elif dtype is str:
            dt = np.dtype([("data", np.dtype("S10")), ("extra", "i8")])
            data = np.array([("hey", 0)] * np.prod(shape), dtype=dt).reshape(shape)
        elif dtype is datetime:
            dt = np.dtype([("data", np.dtype("S32")), ("extra", "i8")])
            data = np.array(
                [(datetime.now(timezone.utc).isoformat().encode("utf-8"), 0)]
                * np.prod(shape),
                dtype=dt,
            ).reshape(shape)
        else:
            dt = np.dtype([("data", dtype), ("extra", "i8")])
            data = np.zeros(shape, dtype=dt)
        h5path = H5ArrayPath(hdf5_file, array_path, "data")

        with h5py.File(hdf5_file, "w") as h5f:
            _ = h5f.create_dataset(array_path, data=data)
        return h5path


class DaskCase(InterfaceCase):
    """In-memory dask array"""

    interface = DaskInterface

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> da.Array:
        if array is not None:
            return da.array(array, dtype=dtype)
        if issubclass(dtype, BaseModel):
            return da.full(shape=shape, fill_value=dtype(x=1), chunks=-1)
        else:
            return da.zeros(shape=shape, dtype=dtype, chunks=10)


class _ZarrMetaCase(InterfaceCase):
    """Shared classmethods for zarr cases"""

    interface = ZarrInterface

    @classmethod
    def skip(cls, shape: Tuple[int, ...], dtype: DtypeType) -> bool:
        return issubclass(dtype, BaseModel) or dtype is str


class ZarrCase(_ZarrMetaCase):
    """In-memory zarr array"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[zarr.Array]:
        if array is not None:
            return zarr.array(array, dtype=dtype, chunks=-1)
        else:
            return zarr.zeros(shape=shape, dtype=dtype)


class ZarrDirCase(_ZarrMetaCase):
    """On-disk zarr array"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[zarr.Array]:
        store = zarr.DirectoryStore(str(path / "array.zarr"))
        if array is not None:
            return zarr.array(array, dtype=dtype, store=store, chunks=-1)
        else:
            return zarr.zeros(shape=shape, dtype=dtype, store=store)


class ZarrZipCase(_ZarrMetaCase):
    """Zarr zip store"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[zarr.Array]:
        store = zarr.ZipStore(str(path / "array.zarr"), mode="w")
        if array is not None:
            return zarr.array(array, dtype=dtype, store=store, chunks=-1)
        else:
            return zarr.zeros(shape=shape, dtype=dtype, store=store)


class ZarrNestedCase(_ZarrMetaCase):
    """Nested zarr array"""

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10),
        dtype: DtypeType = float,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> ZarrArrayPath:
        file = str(path / "nested.zarr")
        root = zarr.open(file, mode="w")
        subpath = "a/b/c"
        if array is not None:
            _ = root.array(subpath, array, dtype=dtype)
        else:
            _ = root.zeros(subpath, shape=shape, dtype=dtype)
        return ZarrArrayPath(file=file, path=subpath)


class VideoCase(InterfaceCase):
    """AVI video"""

    interface = VideoInterface

    @classmethod
    def make_array(
        cls,
        shape: Tuple[int, ...] = (10, 10, 10, 3),
        dtype: DtypeType = np.uint8,
        path: Optional[Path] = None,
        array: Optional[NDArrayType] = None,
    ) -> Optional[Path]:
        if cls.skip(shape, dtype):  # pragma: no cover
            return None

        if array is not None:
            array = np.array(array, dtype=np.uint8)
            shape = array.shape

        is_color = len(shape) == 4
        frames = shape[0]
        frame_shape = shape[1:]

        video_path = path / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"RGBA"),  # raw video for testing purposes
            30,
            (frame_shape[1], frame_shape[0]),
            is_color,
        )
        for i in range(frames):
            if array is not None:
                frame = array[i]
            else:
                # make fresh array every time bc opencv eats them
                frame = np.full(frame_shape, fill_value=i, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    @classmethod
    def skip(cls, shape: Tuple[int, ...], dtype: DtypeType) -> bool:
        """
        We really can only handle 4 dimensional cases in 8-bit rn lol

        .. todo::

            Fix shape/writing for grayscale videos

        """
        if len(shape) != 4:
            return True

        # if len(shape) < 3 or len(shape) > 4:
        #     return True
        if dtype not in (int, np.uint8):
            return True
        # if we have a color video (ie. shape == 4, needs to be RGB)
        if len(shape) == 4 and shape[3] != 3:
            return True
