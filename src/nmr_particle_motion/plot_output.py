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
    VideoContext,
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


def get_frame_to_sec_formatter(config: Config) -> FuncFormatter:
    """Formatter that converts frame indices to seconds for the given video."""

    def frame_to_sec_formatter(x, pos):
        return f"{int(frame_to_time(x, config))}"

    return FuncFormatter(frame_to_sec_formatter)


def plot_height_over_time(
    quant_path: pathlib.Path,
    context: VideoContext,
    config: Config,
    ax: Axes,
):
    """Plot the leading (blue) and lagging (red) edge height in the tube over time."""
    SG = ShapeGlobals()
    nframes = time_to_frames(config.num_secs_to_process, config)
    df = pd.read_csv(quant_path)
    ax.plot(
        df["seconds"][:nframes],
        df["leading_edge_mm"][:nframes],
        label=f"{context.details.lot} - {context.details.vial_number}",
        c="blue",
    )
    ax.plot(df["seconds"][:nframes], df["lagging_edge_mm"][:nframes], c="red")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Height (mm)")
    ax.set_ylim(0, get_mm_distance(SG.FINAL_FRAME_SHAPE[1], config))
    ax.set_title(
        "Particle Height Over Time"
        f"\n{context.details.lot} {context.details.vial_number}, "
    )


def plot_distribution(
    ctx: VideoContext,
    config: Config,
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
    assert ctx.norm_path.exists(), f"Normalized video {ctx.norm_path} does not exist."
    for i, normalized_frame in enumerate(generate_grayscale_frames(ctx.norm_path)):
        t = frame_to_time(i, config)
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
        time_of_interest_str = (
            f"{config.time_of_interest_sec}±{time_margin}"
            if time_margin > 0
            else f"{config.time_of_interest_sec}"
        )
        ax.set_title(
            f"Particle Distribution at {time_of_interest_str} seconds"
            f"\nLot: {ctx.details.lot}, Vial: {ctx.details.vial_number}"
        )
        ax.set_xlabel("Position from lagging to leading edge (%)")
        ax.set_ylabel("Particle Density")
    return


def plot_distribution_by_time(ctx: VideoContext, config: Config, ax: Axes):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    SG = ShapeGlobals()
    nframes = time_to_frames(config.num_secs_to_process, config)
    distributions = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.float32)
    counts = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.int32)
    for i, normalized_frame in enumerate(generate_grayscale_frames(ctx.norm_path)):
        if i >= nframes:
            break
        xx, y, lage, le = get_particle_distribution_from_frame(normalized_frame)
        if xx.size == 0 or y.size == 0:
            continue
        counts[:, i] = normalized_frame.sum(axis=0)
        distributions[lage:le, i] = 255 * y / y.max()
    ax.imshow(distributions, aspect="equal", cmap="gray")
    ax.set_title(
        f"Particle Distribution"
        f"\nLot: {ctx.details.lot}, Vial: {ctx.details.vial_number}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(np.arange(0, nframes, time_to_frames(60, config)))
    ax.xaxis.set_major_formatter(get_frame_to_sec_formatter(config))
    ax.set_ylabel("Height (mm)")
    ax.set_yticks(
        [0, distributions.shape[0]],
        ["0", f"{int(get_mm_distance(distributions.shape[0], config))}"],
    )
    ax.invert_yaxis()
    return distributions


def plot_video_as_image(context: VideoContext, config: Config, ax: Axes):
    """Creates an image where each column
    shows the particle distribution within the tube at that time.
    """
    SG = ShapeGlobals()
    normalizer = VideoNormalizer(context, config)
    nframes = time_to_frames(config.num_secs_to_process, config)
    img = np.zeros((SG.FINAL_FRAME_SHAPE[1], nframes), dtype=np.uint8)
    for i, frame in tqdm(generate_grayscale_frames(context.path), total=nframes):
        if i == 0:
            normalizer.tune_brightness_scale_factor(frame)
        if i >= nframes:
            break
        cropped_frame = normalizer.crop_frame_to_tube(frame)
        blurred = uniform_filter(cropped_frame, size=3)
        scaled = normalizer.scale_brightness(blurred)
        img[:, i] = scaled.mean(axis=0, dtype=np.uint8)
    ax.imshow(img, aspect="equal", cmap="gray")
    ax.set_title(
        f"Frame Averages Over Time"
        f"Lot: {context.details.lot}"
        f"Vial: {context.details.vial_number}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_xticks(np.arange(0, nframes, time_to_frames(60, config)))
    ax.xaxis.set_major_formatter(get_frame_to_sec_formatter(config))
    ax.set_ylabel("Height (mm)")
    ax.set_yticks(
        [0, img.shape[0]], ["0", f"{int(get_mm_distance(img.shape[0], config))}"]
    )
    ax.invert_yaxis()
    return img
