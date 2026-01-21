"""
Frame generation utilities for particle analysis.
"""

import pathlib
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


@dataclass
class VideoData:
    """Data class to hold video and its metadata."""

    video: cv2.VideoCapture
    video_path: pathlib.Path
    nframes: int
    fps: float

    @classmethod
    def from_path(cls, video_path: pathlib.Path) -> "VideoData":
        """Create VideoData from a video file path.
        The video is opened and metadata is extracted.
        The video object is not released here; it should be released by the caller.
        """
        video = cv2.VideoCapture(video_path.as_posix())
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        return cls(video=video, video_path=video_path, nframes=nframes, fps=fps)


def generate_grayscale_frames(
    path_to_video: pathlib.Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Generate frames from the video file.
    Yields the frame index and the frame itself as a numpy array.
    """
    video = cv2.VideoCapture(path_to_video.as_posix())
    vd = VideoData(
        video=video,
        video_path=path_to_video,
        nframes=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=video.get(cv2.CAP_PROP_FPS),
    )
    for i in range(vd.nframes):
        not_at_end, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Force grayscale
        # Check if we've reached the end of the video
        if not_at_end:
            yield i, np.asarray(frame)
        else:
            break
    # Release the video object
    vd.video.release()
