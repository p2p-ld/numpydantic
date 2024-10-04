from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
from numpydantic.testing.helpers import InterfaceCase, ValidationCase


class NumpyCase(InterfaceCase):
    """In-memory numpy array"""

    interface = NumpyInterface

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> np.ndarray:
        if issubclass(case.dtype, BaseModel):
            return np.full(shape=case.shape, fill_value=case.dtype(x=1))
        else:
            return np.zeros(shape=case.shape, dtype=case.dtype)


class _HDF5MetaCase(InterfaceCase):
    """Base case for hdf5 cases"""

    interface = H5Interface

    @classmethod
    def skip(cls, case: "ValidationCase") -> bool:
        return not issubclass(case.dtype, BaseModel)


class HDF5Case(_HDF5MetaCase):
    """HDF5 Array"""

    @classmethod
    def generate_array(
        cls, case: "ValidationCase", path: Path
    ) -> Optional[H5ArrayPath]:
        if cls.skip(case):
            return None

        hdf5_file = path / "h5f.h5"
        array_path = (
            "/" + "_".join([str(s) for s in case.shape]) + "__" + case.dtype.__name__
        )
        generator = np.random.default_rng()

        if case.dtype is str:
            data = generator.random(case.shape).astype(bytes)
        elif case.dtype is datetime:
            data = np.empty(case.shape, dtype="S32")
            data.fill(datetime.now(timezone.utc).isoformat().encode("utf-8"))
        else:
            data = generator.random(case.shape).astype(case.dtype)

        h5path = H5ArrayPath(hdf5_file, array_path)

        with h5py.File(hdf5_file, "w") as h5f:
            _ = h5f.create_dataset(array_path, data=data)
        return h5path


class HDF5CompoundCase(_HDF5MetaCase):
    """HDF5 Array with a fake compound dtype"""

    @classmethod
    def generate_array(
        cls, case: "ValidationCase", path: Path
    ) -> Optional[H5ArrayPath]:
        if cls.skip(case):
            return None

        hdf5_file = path / "h5f.h5"
        array_path = (
            "/" + "_".join([str(s) for s in case.shape]) + "__" + case.dtype.__name__
        )
        if case.dtype is str:
            dt = np.dtype([("data", np.dtype("S10")), ("extra", "i8")])
            data = np.array([("hey", 0)] * np.prod(case.shape), dtype=dt).reshape(
                case.shape
            )
        elif case.dtype is datetime:
            dt = np.dtype([("data", np.dtype("S32")), ("extra", "i8")])
            data = np.array(
                [(datetime.now(timezone.utc).isoformat().encode("utf-8"), 0)]
                * np.prod(case.shape),
                dtype=dt,
            ).reshape(case.shape)
        else:
            dt = np.dtype([("data", case.dtype), ("extra", "i8")])
            data = np.zeros(case.shape, dtype=dt)
        h5path = H5ArrayPath(hdf5_file, array_path, "data")

        with h5py.File(hdf5_file, "w") as h5f:
            _ = h5f.create_dataset(array_path, data=data)
        return h5path


class DaskCase(InterfaceCase):
    """In-memory dask array"""

    interface = DaskInterface

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> da.Array:
        if issubclass(case.dtype, BaseModel):
            return da.full(shape=case.shape, fill_value=case.dtype(x=1), chunks=-1)
        else:
            return da.zeros(shape=case.shape, dtype=case.dtype, chunks=10)


class _ZarrMetaCase(InterfaceCase):
    """Shared classmethods for zarr cases"""

    interface = ZarrInterface

    @classmethod
    def skip(cls, case: "ValidationCase") -> bool:
        return not issubclass(case.dtype, BaseModel)


class ZarrCase(_ZarrMetaCase):
    """In-memory zarr array"""

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> Optional[zarr.Array]:
        return zarr.zeros(shape=case.shape, dtype=case.dtype)


class ZarrDirCase(_ZarrMetaCase):
    """On-disk zarr array"""

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> ZarrArrayPath:
        store = zarr.DirectoryStore(str(path / "array.zarr"))
        return zarr.zeros(shape=case.shape, dtype=case.dtype, store=store)


class ZarrZipCase(_ZarrMetaCase):
    """Zarr zip store"""

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> ZarrArrayPath:
        store = zarr.ZipStore(str(path / "array.zarr"), mode="w")
        return zarr.zeros(shape=case.shape, dtype=case.dtype, store=store)


class ZarrNestedCase(_ZarrMetaCase):
    """Nested zarr array"""

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> ZarrArrayPath:
        file = str(path / "nested.zarr")
        root = zarr.open(file, mode="w")
        subpath = "a/b/c"
        _ = root.zeros(subpath, shape=case.shape, dtype=case.dtype)
        return ZarrArrayPath(file=file, path=subpath)


class VideoCase(InterfaceCase):
    """AVI video"""

    interface = VideoInterface

    @classmethod
    def generate_array(cls, case: "ValidationCase", path: Path) -> Optional[Path]:
        if cls.skip(case):
            return None

        is_color = len(case.shape) == 4
        frames = case.shape[0]
        frame_shape = case.shape[1:]

        video_path = path / "test.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"RGBA"),  # raw video for testing purposes
            30,
            (frame_shape[1], frame_shape[0]),
            is_color,
        )

        for i in range(frames):
            # make fresh array every time bc opencv eats them
            array = np.zeros(frame_shape, dtype=np.uint8)
            if not is_color:
                array[i, i] = i
            else:
                array[i, i, :] = i
            writer.write(array)
        writer.release()
        return video_path

    @classmethod
    def skip(cls, case: "ValidationCase") -> bool:
        """We really can only handle 3-4 dimensional cases in 8-bit rn lol"""
        if len(case.shape) < 3 or len(case.shape) > 4:
            return True
        if case.dtype not in (int, np.uint8):
            return True
        # if we have a color video (ie. shape == 4, needs to be RGB)
        if len(case.shape) == 4 and case.shape[3] != 3:
            return True
