"""Unit conversion functions for particle analysis."""

from nmr_particle_motion.config import Config


def frame_to_time(frame_index: int, config: Config) -> float:
    """Get the time in seconds for the given frame index."""
    return frame_index / config.frames_per_real_sec


def time_to_frames(time_sec: int, config: Config) -> int:
    """Convert time in seconds to number of frames."""
    return int(time_sec * config.frames_per_real_sec)


def get_mm_distance(px_distance: float | int, config: Config) -> float:
    """Convert pixel distance to mm distance."""
    return px_distance / config.px_per_mm
