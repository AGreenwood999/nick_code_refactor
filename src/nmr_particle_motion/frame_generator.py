"""
Frame generation utilities for particle analysis.
"""

import pathlib
import logging
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


logger = logging.getLogger("nmr_particle_motion")


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
    video_path: pathlib.Path,
) -> Generator[np.ndarray, None, None]:
    """Generate frames from the video file.
    Yields the frame index and the frame itself as a numpy array.
    """
    vid_cap = cv2.VideoCapture(video_path)
    while True:
        not_at_end, frame = vid_cap.read()

        if frame is None:
            logger.debug(f"Unable to convert frame of video {video_path} to grayscale.")

        if frame is not None and not_at_end:
            yield np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    vid_cap.release()
