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
    path_to_video: pathlib.Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Generate frames from the video file.
    Yields the frame index and the frame itself as a numpy array.
    """
    vd = VideoData.from_path(path_to_video)
    for i in range(vd.nframes):
        not_at_end, frame = vd.video.read()

        if frame is None:
            logger.debug(
                f"Unable to convert frame {i} of {vd.nframes} of video {vd.video_path} to grayscale."
            )

        if frame is not None and not_at_end:
            yield i, np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            logger.debug(
                f"Converted {i - 1} frames of {vd.nframes} for video {vd.video_path} to grayscale"
            )
            break

    # Release the video object
    vd.video.release()
