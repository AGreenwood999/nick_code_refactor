"""Unit conversion functions for particle analysis."""

import pathlib

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    video_path_to_run_details,
)


def frame_to_time(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], frame_index: int
) -> float:
    """Get the time in seconds for the given frame index."""
    datestamp = (
        video_path_to_run_details(video_path, metadata)
        .datestamp.strip("_")
        .split("_")[0]
    )
    frames_per_rt_secs = config.frames_per_real_sec.get(
        datestamp, config.default_frames_per_real_sec
    )
    return frame_index / frames_per_rt_secs


def time_to_frames(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], time_sec: int
) -> int:
    """Convert time in seconds to number of frames."""
    datestamp = (
        video_path_to_run_details(video_path, metadata)
        .datestamp.strip("_")
        .split("_")[0]
    )
    frames_per_rt_secs = config.frames_per_real_sec.get(
        datestamp, config.default_frames_per_real_sec
    )
    return int(time_sec * frames_per_rt_secs)


def get_mm_distance(config: Config, px_distance: float | int) -> float:
    """Convert pixel distance to mm distance."""
    return px_distance / config.px_per_mm
