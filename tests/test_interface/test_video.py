"""
Needs to be refactored to DRY, but works for now
"""

from pathlib import Path

import cv2
import pytest
from pydantic import BaseModel, ValidationError

from numpydantic import NDArray, Shape
from numpydantic import dtype as dt
from numpydantic.interface.video import VideoProxy

pytestmark = pytest.mark.video


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


@pytest.mark.shape
def test_video_wrong_shape(avi_video):
    shape = (100, 50)

    # generate video with purposely wrong shape
    vid = avi_video(shape=(shape[0] + 10, shape[1] + 10), is_color=True)

    shape_str = f"*, {shape[0]}, {shape[1]}, 3"

    class MyModel(BaseModel):
        array: NDArray[Shape[shape_str], dt.UInt8]

    # should correctly validate :)
    with pytest.raises(ValidationError):
        _ = MyModel(array=vid)


@pytest.mark.proxy
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
    # the fifth frame should be all 5s
    assert (fifth_frame[5, 5, :] == [5, 5, 5]).all()

    # slicing should also work as if it were just a numpy array
    single_slice = instance.array[3, 0:10, 0:5]
    assert single_slice[3, 3, 0] == 3
    assert single_slice.shape == (10, 5, 3)

    # also get a range of frames
    # range without further slices
    range_slice = instance.array[3:5]
    assert range_slice.shape == (2, 100, 50, 3)
    assert range_slice[0, 3, 3, 0] == 3
    assert range_slice[1, 4, 4, 0] == 4

    # full range
    range_slice = instance.array[3:5, 0:10, 0:5]
    assert range_slice.shape == (2, 10, 5, 3)
    assert range_slice[0, 3, 3, 0] == 3
    assert range_slice[1, 4, 4, 0] == 4

    # starting range
    range_slice = instance.array[6:, 0:10, 0:10]
    assert range_slice.shape == (4, 10, 10, 3)
    assert range_slice[-1, 9, 9, 0] == 9
    assert range_slice[-2, 9, 9, 0] == 8

    # ending range
    range_slice = instance.array[:3, 0:5, 0:5]
    assert range_slice.shape == (3, 5, 5, 3)

    # stepped range
    range_slice = instance.array[0:5:2, 0:6, 0:6]
    # second slice should be the second frame (instead of the first)
    assert range_slice.shape == (3, 6, 6, 3)
    assert range_slice[1, 2, 2, 0] == 2
    # and the third should be the fourth (instead of the second)
    assert range_slice[2, 4, 4, 0] == 4

    with pytest.raises(NotImplementedError):
        # shouldn't be allowed to set
        instance.array[5] = 10


@pytest.mark.proxy
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


@pytest.mark.proxy
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


@pytest.mark.proxy
def test_video_not_exists(tmp_path):
    """
    A video file that doesn't exist should raise an error
    """
    video = VideoProxy(tmp_path / "not_real.avi")
    with pytest.raises(FileNotFoundError):
        _ = video.video


@pytest.mark.proxy
@pytest.mark.parametrize(
    "comparison,valid",
    [
        (VideoProxy("test_video.avi"), True),
        (VideoProxy("not_real_video.avi"), False),
        ("not even a video proxy", TypeError),
    ],
)
def test_video_proxy_eq(comparison, valid):
    """
    Comparing a video proxy's equality should be valid if the path matches
    Args:
        comparison:
        valid:

    Returns:

    """
    proxy_a = VideoProxy("test_video.avi")
    if valid is True:
        assert proxy_a == comparison
    elif valid is False:
        assert proxy_a != comparison
    else:
        with pytest.raises(valid):
            assert proxy_a == comparison
