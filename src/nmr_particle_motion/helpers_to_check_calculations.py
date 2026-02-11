"""Helpers to check calculations used in particle analysis."""

import pathlib

from matplotlib import pyplot as plt

from nmr_particle_motion.frame_generator import (
    generate_grayscale_frames,
)
from nmr_particle_motion.shapeglobals import (
    NUM_SECS_TO_PROCESS_IN_VIDEO,
)
from nmr_particle_motion.unit_conversions import (
    frame_to_time,
    time_to_frames,
)
from nmr_particle_motion.video_normalizing import (
    VideoNormalizer,
    plot_frame,
)


def inspect_frame_time_conversion():
    """
    Check whether the frame-to-time and time-to-frame conversions are accurate.
    """
    for run_subdir in RUNS_TO_QUANTIFY[2:]:
        run_dir = ROOT_DIR / run_subdir
        if not run_dir.is_dir():
            continue
        video_path = list(run_dir.glob("*.wmv", case_sensitive=False))[0]
        frames_of_interest = [
            time_to_frames(video_path, t) for t in [0, NUM_SECS_TO_PROCESS_IN_VIDEO]
        ]
        for frame_i, frame in generate_grayscale_frames(video_path):
            if frame_i in frames_of_interest:
                time = frame_to_time(video_path, frame_i)
                plot_frame(frame, f"Frame {frame_i} at time {time:.2f} seconds")
        plt.show()


def inspect_binarization_parameters(vpath: pathlib.Path):
    """Visualize the effect of different binarization parameters on a sample frame."""
    vn = VideoNormalizer(vpath)
    for frame_i, frame in generate_grayscale_frames(vpath):
        if frame_i == time_to_frames(vpath, 0):
            vn.tune_binarization_parameters(frame, show=True)
            break


if __name__ == "__main__":
    inspect_binarization_parameters(
        ROOT_DIR
        / "20251209 Runs"
        / "3E1 2000ug 100rpm d2___2025-12-09_15-01-29-4307-06-00.wmv"
    )
    inspect_binarization_parameters(
        ROOT_DIR / "20251215 Runs" / "Settle___2025-12-15_13-14-14-1649-06-00.wmv"
    )
