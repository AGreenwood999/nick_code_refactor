"""Unit conversion functions for particle analysis."""

import pathlib

from nmr_particle_motion.particle_analysis_lib.file_names_and_paths import (
    video_path_to_run_details,
)
from nmr_particle_motion.particle_analysis_lib.globals import (
    DEFAULT_FRAMES_PER_REAL_TIME_SECONDS,
    PX_PER_MM,
    VIDEO_FRAMES_PER_REAL_TIME_SECONDS,
)


def frame_to_time(video_path: pathlib.Path, frame_index: int) -> float:
    """Get the time in seconds for the given frame index."""
    datestamp = video_path_to_run_details(video_path).datestamp.strip("_").split("_")[0]
    frames_per_rt_secs = VIDEO_FRAMES_PER_REAL_TIME_SECONDS.get(
        datestamp, DEFAULT_FRAMES_PER_REAL_TIME_SECONDS
    )
    return frame_index / frames_per_rt_secs


def time_to_frames(video_path: pathlib.Path, time_sec: int) -> int:
    """Convert time in seconds to number of frames."""
    datestamp = video_path_to_run_details(video_path).datestamp.strip("_").split("_")[0]
    frames_per_rt_secs = VIDEO_FRAMES_PER_REAL_TIME_SECONDS.get(
        datestamp, DEFAULT_FRAMES_PER_REAL_TIME_SECONDS
    )
    return int(time_sec * frames_per_rt_secs)


def get_mm_distance(px_distance: float | int) -> float:
    """Convert pixel distance to mm distance."""
    return px_distance / PX_PER_MM
