"""
Module contains all functions that generate plots.
"""

import pathlib

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    generate_video_paths,
    get_le_lage_fpath,
    get_normalized_video_fpath,
    get_prenormalized_video_fpath,
    video_path_to_run_details,
)
from nmr_particle_motion.frame_generator import generate_grayscale_frames
from nmr_particle_motion.leading_lagging_edge_and_particle_dist import (
    get_particle_distribution_from_frame,
)
from nmr_particle_motion.shapeglobals import ShapeGlobals
from nmr_particle_motion.unit_conversions import (
    frame_to_time,
    get_mm_distance,
    time_to_frames,
)
from nmr_particle_motion.video_normalizing import (
    VideoNormalizer,
    uniform_filter,
)


def get_frame_to_sec_formatter(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> FuncFormatter:
    """Formatter that converts frame indices to seconds for the given video."""

    def frame_to_sec_formatter(x, pos):
        return f"{int(frame_to_time(video_path, config, metadata, x))}"

    return FuncFormatter(frame_to_sec_formatter)


def plot_height_over_time(
    quant_path: pathlib.Path,
    config: Config,
    metadata: dict[str, str],
    ax: Axes,
    title_includes_magnet_distance=True,
):
    """Plot the leading (blue) and lagging (red) edge height in the tube over time."""
    vd = video_path_to_run_details(quant_path, metadata)
    SG = ShapeGlobals()
    nframes = time_to_frames(quant_path, config, metadata, config.num_secs_to_process)
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
    ax.set_ylim(0, get_mm_distance(config, SG.FINAL_FRAME_SHAPE[1]))
    if title_includes_magnet_distance:
        ax.set_title(
            "Particle Height Over Time"
            f"\n{vd.mass} {vd.substance}, "
            f"Magnet {config.magnet_distance_mm[vd.distance]}mm away at {vd.rpm}"
        )
    else:
        ax.set_title(f"{vd.mass} {vd.substance}, Magnet at {vd.rpm}")


def plot_height_at_time_of_interest_by_distance(
    path_to_videos: str | pathlib.Path,
    config: Config,
    metadata: dict[str, str],
    substance,
    mass,
    rpm,
    ax: Axes,
    time_margin=5,
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
    for vpath, vd, _ in generate_video_paths(path_to_videos, config):
        SG = ShapeGlobals()
        if vd.substance != substance or vd.mass != mass or vd.rpm != rpm:
            continue
        quant_path = get_le_lage_fpath(vpath, config, metadata)
        df = pd.read_csv(quant_path)
        frame_without_margin = int(
            time_to_frames(quant_path, config, metadata, config.time_of_interest_sec)
        )
        frame_range = [
            int(
                time_to_frames(
                    quant_path,
                    config,
                    metadata,
                    config.time_of_interest_sec - time_margin,
                )
            ),
            int(
                time_to_frames(
                    quant_path,
                    config,
                    metadata,
                    config.time_of_interest_sec + time_margin,
                )
            ),
        ]
        details = video_path_to_run_details(quant_path, metadata)
        distances.append(config.magnet_distance_mm[details.distance])
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
    ax.set_ylabel(
        f"Height of leading edge at {config.time_of_interest_sec}±{time_margin}s (mm)"
    )
    time_of_interest_str = (
        f"{config.time_of_interest_sec}±{time_margin}"
        if time_margin > 0
        else f"{config.time_of_interest_sec}"
    )
    ax.set_title(
        f"Height at {time_of_interest_str} seconds\n{mass} {substance}, Magnet at {rpm}"
    )
    if SG:
        ax.set_ylim(0, get_mm_distance(config, SG.FINAL_FRAME_SHAPE[1]) + 10)
    ax.set_xticks(sorted(list(config.magnet_distance_mm.values())))
    ax.legend()


def plot_distribution(
    video_path: pathlib.Path,
    config: Config,
    metadata: dict[str, str],
    time_sec: int,
    ax: Axes,
    time_margin=5,
) -> None:
    """Plot the particle distribution at the given time in seconds.
    Width is normalized to 100% from lagging to leading edge.

    0 time_margin means exact time only. Non-zero margin instructs
    to take the non-empty distribution closest to the middle of the window.
    """
    xxs, ys = [], []
    norm_video_path = get_normalized_video_fpath(video_path, config, metadata)
    assert norm_video_path.exists(), (
        f"Normalized video {norm_video_path} does not exist."
    )
    for i, normalized_frame in generate_grayscale_frames(norm_video_path):
        t = frame_to_time(video_path, config, metadata, i)
        if t < (time_sec - time_margin):
            continue
        if t > (time_sec + time_margin):
            break
        xx, y, _, _ = get_particle_distribution_from_frame(normalized_frame)
        if xx.size == 0 or y.size == 0:
            continue
        xxs.append(xx)
        ys.append(y)
    if xxs:
        i = int(len(xxs) / 2)
        xx, y = xxs[i], ys[i]
        ax.plot(xx, y, label="Particle Density")
        vd = video_path_to_run_details(video_path, metadata)
        time_of_interest_str = (
            f"{config.time_of_interest_sec}±{time_margin}"
            if time_margin > 0
            else f"{config.time_of_interest_sec}"
        )
        ax.set_title(
            f"Particle Distribution at {time_of_interest_str} seconds"
            f"\n{vd.mass} {vd.substance}, Magnet {config.magnet_distance_mm[vd.distance]}mm away at {vd.rpm}"
        )
        ax.set_xlabel("Position from lagging to leading edge (%)")
        ax.set_ylabel("Particle Density")
    return


def plot_distribution_by_time(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], ax: Axes
):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    SG = ShapeGlobals()
    nframes = time_to_frames(video_path, config, metadata, config.num_secs_to_process)
    distributions = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.float32)
    counts = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.int32)
    norm_video_path = get_normalized_video_fpath(video_path, config, metadata)
    for i, normalized_frame in generate_grayscale_frames(norm_video_path):
        if i >= nframes:
            break
        xx, y, lage, le = get_particle_distribution_from_frame(
            normalized_frame, xx_as_percents=False
        )
        if xx.size == 0 or y.size == 0:
            continue
        counts[:, i] = normalized_frame.sum(axis=0)
        distributions[lage:le, i] = 255 * y / y.max()
    ax.imshow(distributions, aspect="equal", cmap="gray")
    vd = video_path_to_run_details(video_path, metadata)
    ax.set_title(
        f"Particle Distribution"
        f"\n{vd.mass} {vd.substance}, Magnet "
        f"{config.magnet_distance_mm[vd.distance]}mm away at {vd.rpm}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(
        np.arange(0, nframes, time_to_frames(video_path, config, metadata, 60))
    )
    ax.xaxis.set_major_formatter(
        get_frame_to_sec_formatter(video_path, config, metadata)
    )
    ax.set_ylabel("Height (mm)")
    ax.set_yticks(
        [0, distributions.shape[0]],
        ["0", f"{int(get_mm_distance(config, distributions.shape[0]))}"],
    )
    ax.invert_yaxis()
    return distributions


def plot_video_as_image(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], ax: Axes
):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    video_path = get_prenormalized_video_fpath(video_path, metadata)
    SG = ShapeGlobals()
    normalizer = VideoNormalizer(video_path, config, metadata)
    nframes = time_to_frames(video_path, config, metadata, config.num_secs_to_process)
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
    vd = video_path_to_run_details(video_path, metadata)
    ax.set_title(
        f"Frame Averages Over Time"
        f"\n{vd.mass} {vd.substance}, Magnet "
        f"{config.magnet_distance_mm[vd.distance]}mm away at {vd.rpm}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(
        np.arange(0, nframes, time_to_frames(video_path, config, metadata, 60))
    )
    ax.xaxis.set_major_formatter(
        get_frame_to_sec_formatter(video_path, config, metadata)
    )
    ax.set_ylabel("Height (mm)")
    ax.set_yticks(
        [0, img.shape[0]], ["0", f"{int(get_mm_distance(config, img.shape[0]))}"]
    )
    ax.invert_yaxis()
    return img
