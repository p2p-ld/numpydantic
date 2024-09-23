"""
Interface to support treating videos like arrays using OpenCV
"""

from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from pydantic_core.core_schema import SerializationInfo

from numpydantic.interface import JsonDict
from numpydantic.interface.interface import Interface

try:
    import cv2
    from cv2 import VideoCapture
except ImportError:  # pragma: no cover
    cv2 = None
    VideoCapture = None

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class VideoJsonDict(JsonDict):
    """Json-able roundtrip representation of a video file"""

    type: Literal["video"]
    file: str

    def to_array_input(self) -> "VideoProxy":
        """
        Construct a :class:`.VideoProxy`
        """
        return VideoProxy(path=Path(self.file))


class VideoProxy:
    """
    Passthrough proxy class to interact with videos as arrays
    """

    def __init__(
        self, path: Optional[Path] = None, video: Optional[VideoCapture] = None
    ):
        if path is None and video is None:  # pragma: no cover
            raise ValueError(
                "Need to either supply a path or an opened VideoCapture object"
            )

        if path is not None:
            path = Path(path).resolve()
        self.path = path

        self._video = video  # type: Optional[VideoCapture]
        self._n_frames = None  # type: Optional[int]
        self._dtype = None  # type: Optional[np.dtype]
        self._shape = None  # type: Optional[Tuple[int, ...]]
        self._sample_frame = None  # type: Optional[np.ndarray]

    @property
    def video(self) -> VideoCapture:
        """Opened video capture object"""
        if self._video is None:
            if self.path is None:  # pragma: no cover
                raise RuntimeError(
                    "Instantiated with a VideoCapture object that has been closed, "
                    "and it cant be reopened since source path cant be gotten "
                    "from VideoCapture objects"
                )
            if not self.path.exists():
                raise FileNotFoundError(f"Video file {self.path} does not exist!")

            self._video = VideoCapture(str(self.path))
        return self._video

    def close(self) -> None:
        """Close the opened VideoCapture object"""
        if self._video is not None:
            self._video.release()
            self._video = None

    @property
    def sample_frame(self) -> np.ndarray:
        """A stored frame from the video to use when calculating shape and dtype"""
        if self._sample_frame is None:
            current_frame = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))

            self.video.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 1))
            status, frame = self.video.read()
            if not status:  # pragma: no cover
                raise RuntimeError("Could not read frame from video")
            self.video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            self._sample_frame = frame
        return self._sample_frame

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of video like
        ``(n_frames, height, width, channels)``

        Note that this order flips the order of height and width from typical resolution
        specifications: eg. 1080p video is typically 1920x1080, but here it would be
        1080x1920. This follows opencv's ordering, which matches expectations when
        eg. an image is read and plotted with matplotlib: the first index is the
        position in the 0th dimension - the height, or "y" axis - and the second is the
        width/x.
        """
        if self._shape is None:
            self._shape = (self.n_frames, *self.sample_frame.shape)
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype (from ``sample_frame`` )"""
        return self.sample_frame.dtype

    @property
    def n_frames(self) -> int:
        """
        Try to get number of frames using opencv metadata, and manually count if no
        t"""
        if self._n_frames is None:
            n_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
            if n_frames == 0:  # pragma: no cover
                # have to count manually for some containers with bad metadata
                # not testing for now, will wait until we encounter such a
                # video in the wild where this doesn't work.
                current_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                n_frames = 0
                while True:
                    status, _ = self.video.read()
                    if not status:
                        break
                    n_frames += 1
                self.video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            self._n_frames = int(n_frames)
        return self._n_frames

    def _get_frame(self, frame: int) -> np.ndarray:
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        status, frame = self.video.read()
        if not status:  # pragma: no cover
            raise ValueError(f"Could not get frame {frame}")
        return frame

    def _complete_slice(self, slice_: slice) -> slice:
        """Get a fully-built slice that can be passed to range"""
        if slice_.step is None:
            slice_ = slice(slice_.start, slice_.stop, 1)
        if slice_.stop is None:
            slice_ = slice(slice_.start, self.n_frames, slice_.step)
        if slice_.start is None:
            slice_ = slice(0, slice_.stop, slice_.step)
        return slice_

    def __array__(self) -> np.ndarray:
        """Whole video as a numpy array"""
        return self[:]

    def __getitem__(self, item: Union[int, slice, tuple]) -> np.ndarray:
        if isinstance(item, int):
            # want a single frame
            return self._get_frame(item)
        elif isinstance(item, slice):
            # slice of frames
            item = self._complete_slice(item)
            frames = [
                self._get_frame(i) for i in range(item.start, item.stop, item.step)
            ]
            return np.stack(frames)
        else:
            # slices are passed as tuples
            # first arg needs to be handled specially
            if isinstance(item[0], int):
                # single frame
                frame = self._get_frame(item[0])
                # syntax doesn't work in 3.9 but would be more explicit...
                # return frame[*item[1:]]
                return frame[item[1:]]

            elif isinstance(item[0], slice):
                frames = []
                # make a new slice since range cant take Nones, filling in missing vals
                fslice = self._complete_slice(item[0])

                for i in range(fslice.start, fslice.stop, fslice.step):
                    frames.append(self._get_frame(i))
                frame = np.stack(frames)
                # syntax doesn't work in 3.9 but would be simpler..
                # return frame[:, *item[1:]]
                # construct a new slice instead
                new_slice = (slice(None, None, None), *item[1:])
                return frame[new_slice]
            else:  # pragma: no cover
                raise ValueError(f"indices must be an int or a slice! got {item}")

    def __setitem__(self, key: Union[int, slice], value: Union[int, float, np.ndarray]):
        raise NotImplementedError("Setting pixel values on videos is not supported!")

    def __getattr__(self, item: str):
        if item == "__name__":
            return "VideoProxy"
        return getattr(self.video, item)

    def __eq__(self, other: "VideoProxy") -> bool:
        """Check if this is a proxy to the same video file"""
        if not isinstance(other, VideoProxy):
            raise TypeError("Can only compare equality of two VideoProxies")
        return self.path == other.path

    def __len__(self) -> int:
        """Number of frames in the video"""
        return self.shape[0]


class VideoInterface(Interface):
    """
    OpenCV interface to treat videos as arrays.
    """

    name = "video"
    input_types = (str, Path, VideoCapture, VideoProxy)
    return_type = VideoProxy
    json_model = VideoJsonDict

    @classmethod
    def enabled(cls) -> bool:
        """Check if opencv-python is available in the environment"""
        return cv2 is not None

    @classmethod
    def check(cls, array: Any) -> bool:
        """
        Check if array is a string or Path with a supported video extension,
        or an opened VideoCapture object
        """
        if (VideoCapture is not None and isinstance(array, VideoCapture)) or isinstance(
            array, VideoProxy
        ):
            return True

        if isinstance(array, dict):
            array = array.get("file", "")

        if isinstance(array, str):
            try:
                array = Path(array)
            except TypeError:  # pragma: no cover
                # fine, just not a video
                return False

        return isinstance(array, Path) and array.suffix.lower() in VIDEO_EXTENSIONS

    def before_validation(self, array: Any) -> VideoProxy:
        """Get a :class:`.VideoProxy` object for this video"""
        if isinstance(array, VideoCapture):
            proxy = VideoProxy(video=array)
        elif isinstance(array, VideoProxy):
            proxy = array
        else:
            proxy = VideoProxy(path=array)
        return proxy

    @classmethod
    def to_json(
        cls, array: VideoProxy, info: SerializationInfo
    ) -> Union[list, VideoJsonDict]:
        """Return a json-representation of a video"""
        if info.round_trip:
            return VideoJsonDict(type=cls.name, file=str(array.path))
        else:
            return np.array(array).tolist()
