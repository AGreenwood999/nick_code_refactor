"""
Module contains all functions that generate plots.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from nmr_particle_motion.particle_analysis_lib.file_names_and_paths import (
    generate_video_paths,
    get_le_lage_fpath,
    get_normalized_video_fpath,
    get_prenormalized_video_fpath,
    video_has_mm_scale,
    video_path_to_run_details,
)
from nmr_particle_motion.particle_analysis_lib.frame_generator import (
    generate_grayscale_frames,
)
from nmr_particle_motion.particle_analysis_lib.globals import (
    MAGNET_DISTANCE_TO_MM,
    NUM_SECS_TO_PROCESS_IN_VIDEO,
    SAVE_DIR,
    TIME_OF_INTEREST,
    ShapeGlobals,
)
from nmr_particle_motion.particle_analysis_lib.leading_lagging_edge_and_particle_dist import (
    get_particle_distribution_from_frame,
)
from nmr_particle_motion.particle_analysis_lib.unit_conversions import (
    frame_to_time,
    get_mm_distance,
    time_to_frames,
)
from nmr_particle_motion.particle_analysis_lib.video_normalizing import (
    VideoNormalizer,
    uniform_filter,
)


def get_frame_to_sec_formatter(video_path: pathlib.Path) -> FuncFormatter:
    """Formatter that converts frame indices to seconds for the given video."""

    def frame_to_sec_formatter(x, pos):
        return f"{int(frame_to_time(video_path, x))}"

    return FuncFormatter(frame_to_sec_formatter)


def plot_height_over_time(
    quant_path: pathlib.Path, ax: Axes, title_includes_magnet_distance=True
):
    """Plot the leading (blue) and lagging (red) edge height in the tube over time."""
    vd = video_path_to_run_details(quant_path)
    SG = ShapeGlobals(vd.has_mm_scale)
    nframes = time_to_frames(quant_path, NUM_SECS_TO_PROCESS_IN_VIDEO)
    df = pd.read_csv(quant_path)
    ax.plot(
        df["seconds"][:nframes],
        df["leading_edge_mm"][:nframes],
        label=f"{vd.substance} {vd.mass} {vd.rpm} {vd.distance}",
        c="blue",
    )
    ax.plot(df["seconds"][:nframes], df["lagging_edge_mm"][:nframes], c="red")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Height (mm)")
    ax.set_ylim(0, get_mm_distance(SG.FINAL_FRAME_SHAPE[1]))
    if title_includes_magnet_distance:
        ax.set_title(
            "Particle Height Over Time"
            f"\n{vd.mass} {vd.substance}, "
            f"Magnet {MAGNET_DISTANCE_TO_MM[vd.distance]}mm away at {vd.rpm}"
        )
    else:
        ax.set_title(f"{vd.mass} {vd.substance}, Magnet at {vd.rpm}")


def plot_height_at_time_of_interest_by_distance(
    substance,
    mass,
    rpm,
    ax: Axes,
    time_margin=5,
    path_to_videos: str | pathlib.Path | None = None,
):
    """Plot height at TIME_OF_INTEREST seconds vs. distance from magnet.

    This suggests to us the distance from the magnet that maximizes
    the leading and lagging edge of the particles.

    0 time_margin means exact time only. Non-zero margin instructs to take max in that window,
    which is useful because there is jitter in the measurement
    (sometimes particles seem to disappear for a second).
    """
    distances = []
    le_heights = []
    lage_heights = []
    SG = None
    for vpath, vd in generate_video_paths(path_to_videos):
        SG = ShapeGlobals(video_has_mm_scale(vpath))
        if vd.substance != substance or vd.mass != mass or vd.rpm != rpm:
            continue
        quant_path = get_le_lage_fpath(vpath)
        df = pd.read_csv(quant_path)
        frame_without_margin = int(time_to_frames(quant_path, TIME_OF_INTEREST))
        frame_range = [
            int(time_to_frames(quant_path, TIME_OF_INTEREST - time_margin)),
            int(time_to_frames(quant_path, TIME_OF_INTEREST + time_margin)),
        ]
        details = video_path_to_run_details(quant_path)
        distances.append(MAGNET_DISTANCE_TO_MM[details.distance])
        if time_margin > 0:
            le_heights.append(
                df["leading_edge_mm"][frame_range[0] : frame_range[1] + 1].max()
            )
            lage_heights.append(
                df["lagging_edge_mm"][frame_range[0] : frame_range[1] + 1].max()
            )
        else:
            le_heights.append(df["leading_edge_mm"][frame_without_margin])
            lage_heights.append(df["lagging_edge_mm"][frame_without_margin])
    order = np.argsort(distances)
    le_heights = np.asarray(le_heights)[order]  # type: ignore[assignment]
    lage_heights = np.asarray(lage_heights)[order]  # type: ignore[assignment]
    distances = np.asarray(distances)[order]  # type: ignore[assignment]
    ax.plot(distances, le_heights, label="Leading Edge", c="blue", marker="o")
    ax.plot(distances, lage_heights, label="Lagging Edge", c="red", marker="o")
    ax.set_xlabel("Distance from magnet (mm)")
    ax.set_ylabel(f"Height of leading edge at {TIME_OF_INTEREST}±{time_margin}s (mm)")
    time_of_interest_str = (
        f"{TIME_OF_INTEREST}±{time_margin}"
        if time_margin > 0
        else f"{TIME_OF_INTEREST}"
    )
    ax.set_title(
        f"Height at {time_of_interest_str} seconds\n{mass} {substance}, Magnet at {rpm}"
    )
    if SG:
        ax.set_ylim(0, get_mm_distance(SG.FINAL_FRAME_SHAPE[1]) + 10)
    ax.set_xticks(sorted(list(MAGNET_DISTANCE_TO_MM.values())))
    ax.legend()


def plot_distribution(
    video_path: pathlib.Path, time_sec: int, ax: Axes, time_margin=5
) -> None:
    """Plot the particle distribution at the given time in seconds.
    Width is normalized to 100% from lagging to leading edge.

    0 time_margin means exact time only. Non-zero margin instructs
    to take the non-empty distribution closest to the middle of the window.
    """
    xxs, ys = [], []
    norm_video_path = get_normalized_video_fpath(video_path)
    assert norm_video_path.exists(), (
        f"Normalized video {norm_video_path} does not exist."
    )
    for i, normalized_frame in generate_grayscale_frames(norm_video_path):
        t = frame_to_time(video_path, i)
        if t < (time_sec - time_margin):
            continue
        if t > (time_sec + time_margin):
            break
        xx, y, _lage, _le = get_particle_distribution_from_frame(
            normalized_frame, video_has_mm_scale(video_path)
        )
        if xx.size == 0 or y.size == 0:
            continue
        xxs.append(xx)
        ys.append(y)
    if xxs:
        i = int(len(xxs) / 2)
        xx, y = xxs[i], ys[i]
        ax.plot(xx, y, label="Particle Density")
        vd = video_path_to_run_details(video_path)
        time_of_interest_str = (
            f"{TIME_OF_INTEREST}±{time_margin}"
            if time_margin > 0
            else f"{TIME_OF_INTEREST}"
        )
        ax.set_title(
            f"Particle Distribution at {time_of_interest_str} seconds"
            f"\n{vd.mass} {vd.substance}, Magnet {MAGNET_DISTANCE_TO_MM[vd.distance]}mm away at {vd.rpm}"
        )
        ax.set_xlabel("Position from lagging to leading edge (%)")
        ax.set_ylabel("Particle Density")
    return


def plot_distribution_by_time(video_path: pathlib.Path, ax: Axes):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    SG = ShapeGlobals(video_has_mm_scale(video_path))
    nframes = time_to_frames(video_path, NUM_SECS_TO_PROCESS_IN_VIDEO)
    distributions = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.float32)
    counts = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.int32)
    norm_video_path = get_normalized_video_fpath(video_path)
    for i, normalized_frame in generate_grayscale_frames(norm_video_path):
        if i >= nframes:
            break
        xx, y, lage, le = get_particle_distribution_from_frame(
            normalized_frame, SG.has_mm_scale, xx_as_percents=False
        )
        if xx.size == 0 or y.size == 0:
            continue
        counts[:, i] = normalized_frame.sum(axis=0)
        distributions[lage:le, i] = 255 * y / y.max()
    ax.imshow(distributions, aspect="equal", cmap="gray")
    vd = video_path_to_run_details(video_path)
    ax.set_title(
        f"Particle Distribution"
        f"\n{vd.mass} {vd.substance}, Magnet "
        f"{MAGNET_DISTANCE_TO_MM[vd.distance]}mm away at {vd.rpm}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(np.arange(0, nframes, time_to_frames(video_path, 60)))
    ax.xaxis.set_major_formatter(get_frame_to_sec_formatter(video_path))
    ax.set_ylabel("Height (mm)")
    ax.set_yticks(
        [0, distributions.shape[0]],
        ["0", f"{int(get_mm_distance(distributions.shape[0]))}"],
    )
    ax.invert_yaxis()
    return distributions


def plot_video_as_image(video_path: pathlib.Path, ax: Axes):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    video_path = get_prenormalized_video_fpath(video_path)
    SG = ShapeGlobals(video_has_mm_scale(video_path))
    normalizer = VideoNormalizer(video_path)
    nframes = time_to_frames(video_path, NUM_SECS_TO_PROCESS_IN_VIDEO)
    img = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.uint8)
    for i, frame in tqdm(generate_grayscale_frames(video_path), total=nframes):
        if i == 0:
            normalizer.tune_brightness_scale_factor(frame)
        if i >= nframes:
            break
        cropped_frame = normalizer.crop_frame_to_tube(frame)
        blurred = uniform_filter(cropped_frame, size=3)
        scaled = normalizer.scale_brightness(blurred)
        img[:, i] = scaled.mean(axis=0, dtype=np.uint8)
    ax.imshow(img, aspect="equal", cmap="gray")
    vd = video_path_to_run_details(video_path)
    ax.set_title(
        f"Frame Averages Over Time"
        f"\n{vd.mass} {vd.substance}, Magnet "
        f"{MAGNET_DISTANCE_TO_MM[vd.distance]}mm away at {vd.rpm}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(np.arange(0, nframes, time_to_frames(video_path, 60)))
    ax.xaxis.set_major_formatter(get_frame_to_sec_formatter(video_path))
    ax.set_ylabel("Height (mm)")
    ax.set_yticks([0, img.shape[0]], ["0", f"{int(get_mm_distance(img.shape[0]))}"])
    ax.invert_yaxis()
    return img
