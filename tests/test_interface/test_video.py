"""
Needs to be refactored to DRY, but works for now
"""

import pdb

import numpy as np
import pytest

from pathlib import Path
import cv2

from pydantic import BaseModel, ValidationError

from numpydantic import NDArray, Shape
from numpydantic import dtype as dt
from numpydantic.interface.video import VideoProxy


@pytest.fixture(scope="function")
def avi_video(tmp_path):
    video_path = tmp_path / "test.avi"

    def _make_video(shape=(100, 50), frames=10, is_color=True) -> Path:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"RGBA"),  # raw video for testing purposes
            30,
            (shape[1], shape[0]),
            is_color,
        )
        if is_color:
            shape = (*shape, 3)

        for i in range(frames):
            # make fresh array every time bc opencv eats them
            array = np.zeros(shape, dtype=np.uint8)
            if not is_color:
                array[i, i] = i
            else:
                array[i, i, :] = i
            writer.write(array)
        writer.release()
        return video_path

    yield _make_video

    video_path.unlink(missing_ok=True)


@pytest.mark.parametrize("input_type", [str, Path])
def test_video_validation(avi_video, input_type):
    """Color videos should validate for normal uint8 shape specs"""

    shape = (100, 50)
    vid = avi_video(shape=shape, is_color=True)
    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    # should correctly validate :)
    instance = MyModel(array=input_type(vid))
    assert isinstance(instance.array, VideoProxy)


def test_video_from_videocapture(avi_video):
    """Should be able to pass an opened videocapture object"""
    shape = (100, 50)
    vid = avi_video(shape=shape, is_color=True)
    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    # should still correctly validate!
    opened_vid = cv2.VideoCapture(str(vid))
    try:
        instance = MyModel(array=opened_vid)
        assert isinstance(instance.array, VideoProxy)
    finally:
        opened_vid.release()


def test_video_wrong_shape(avi_video):
    shape = (100, 50)

    # generate video with purposely wrong shape
    vid = avi_video(shape=(shape[0] + 10, shape[1] + 10), is_color=True)

    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    # should correctly validate :)
    with pytest.raises(ValidationError):
        instance = MyModel(array=vid)


def test_video_getitem(avi_video):
    """
    Should be able to get individual frames and slices as if it were a normal array
    """
    shape = (100, 50)
    vid = avi_video(shape=shape, frames=10, is_color=True)
    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    instance = MyModel(array=vid)
    fifth_frame = instance.array[5]
    # the first frame should have 1's in the 1,1 position
    assert (fifth_frame[5, 5, :] == [5, 5, 5]).all()
    # and nothing in the 6th position
    assert (fifth_frame[6, 6, :] == [0, 0, 0]).all()

    # slicing should also work as if it were just a numpy array
    single_slice = instance.array[3, 0:10, 0:5]
    assert single_slice[3, 3, 0] == 3
    assert single_slice[4, 4, 0] == 0
    assert single_slice.shape == (10, 5, 3)

    # also get a range of frames
    # range without further slices
    range_slice = instance.array[3:5]
    assert range_slice.shape == (2, 100, 50, 3)
    assert range_slice[0, 3, 3, 0] == 3
    assert range_slice[0, 4, 4, 0] == 0

    # full range
    range_slice = instance.array[3:5, 0:10, 0:5]
    assert range_slice.shape == (2, 10, 5, 3)
    assert range_slice[0, 3, 3, 0] == 3
    assert range_slice[0, 4, 4, 0] == 0

    # starting range
    range_slice = instance.array[6:, 0:10, 0:10]
    assert range_slice.shape == (4, 10, 10, 3)
    assert range_slice[-1, 9, 9, 0] == 9
    assert range_slice[-2, 9, 9, 0] == 0

    # ending range
    range_slice = instance.array[:3, 0:5, 0:5]
    assert range_slice.shape == (3, 5, 5, 3)

    # stepped range
    range_slice = instance.array[0:5:2, 0:6, 0:6]
    # second slice should be the second frame (instead of the first)
    assert range_slice.shape == (3, 6, 6, 3)
    assert range_slice[1, 2, 2, 0] == 2
    assert range_slice[1, 3, 3, 0] == 0
    # and the third should be the fourth (instead of the second)
    assert range_slice[2, 4, 4, 0] == 4
    assert range_slice[2, 5, 5, 0] == 0

    with pytest.raises(NotImplementedError):
        # shouldn't be allowed to set
        instance.array[5] = 10


def test_video_attrs(avi_video):
    """Should be able to access opencv properties"""
    shape = (100, 50)
    vid = avi_video(shape=shape, is_color=True)
    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    instance = MyModel(array=vid)

    instance.array.set(cv2.CAP_PROP_POS_FRAMES, 5)
    assert int(instance.array.get(cv2.CAP_PROP_POS_FRAMES)) == 5


def test_video_close(avi_video):
    """Should close and reopen video file if needed"""
    shape = (100, 50)
    vid = avi_video(shape=shape, is_color=True)
    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    instance = MyModel(array=vid)
    assert isinstance(instance.array.video, cv2.VideoCapture)
    # closes releases and removed reference
    instance.array.close()
    assert instance.array._video is None
    # reopen
    assert isinstance(instance.array.video, cv2.VideoCapture)
