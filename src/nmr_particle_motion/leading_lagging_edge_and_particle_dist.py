"""Module to quantify particle behavior in a video.
It calculated the leading and lagging edge of particles
by finding the rightmost and leftmost white pixels in binary (black-and-white) frames.
"""

import pathlib
from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    get_le_lage_fpath,
    get_normalized_video_fpath,
)
from nmr_particle_motion.frame_generator import (
    generate_grayscale_frames,
)
from nmr_particle_motion.shapeglobals import ShapeGlobals
from nmr_particle_motion.unit_conversions import (
    frame_to_time,
    get_mm_distance,
)
import logging

logger = logging.getLogger("nmr_particle_motion")


def _write_lagging_leading_edges(
    fpath: pathlib.Path,
    config: Config,
    metadata: dict[str, str],
    lage: list[int],
    le: list[int],
) -> None:
    """Write the lagging and leading edge results to a CSV file."""
    with open(fpath, "wt", encoding="utf-8") as ofh:
        ofh.write(
            "frame_index,seconds,lagging_edge_px,leading_edge_px,lagging_edge_mm,leading_edge_mm\n"
        )
        for i, (_lage, _le) in enumerate(zip(lage, le)):
            ofh.write(
                f"{i},{frame_to_time(fpath, config, metadata, i)},{_lage},{_le},{get_mm_distance(config, _lage)},{get_mm_distance(config, _le)}\n"
            )


def _quantify_with_leading_lagging_edges(
    normalized_frame: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Quantify the bead mixing in the current frame.
    This method can be customized to perform specific quantification logic.

    Quantification is the count of white pixels in each column,
    leading edge is the highest row index with a white pixel,
    lagging edge is the lowest row index with a white pixel.
    """
    SG = ShapeGlobals()
    normalized_frame[~SG.QUANT_MASK] = 0
    quant = (normalized_frame > 0).sum(axis=0)
    lagging_edge_px = int(np.argmin(quant < 1))
    leading_edge_px = int(quant.shape[0] - np.argmin(quant[::-1] < 1))
    if quant.sum() == 0:
        lagging_edge_px = 0
        leading_edge_px = 0
    return quant, lagging_edge_px, leading_edge_px


def get_particle_distribution_from_frame(
    normalized_frame: np.ndarray, xx_as_percents=True
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Get the particle distribution from a single normalized frame.
    This method uses a kernel density estimate (KDE) to smooth the particle count distribution.
    By default, the resulting x values are normalized such that it spans the
    breadth of the particle cloud, not necessarily the full length of the tube.

    Parameters
    ----------
    normalized_frame : np.ndarray
        The normalized frame to analyze.
    xx_as_percents : bool, optional
        Whether to return the x values as percents of the visible particle width, by default True.
    """
    xx: np.ndarray = np.asarray([])
    y: np.ndarray = np.asarray([])
    counts, lage, le = _quantify_with_leading_lagging_edges(normalized_frame)
    counts = counts[lage:le]
    if len(counts) == 0 or counts.sum() == 0:
        return xx, y, lage, le
    if counts.sum() < 10:
        xx = np.arange(len(counts))
        if xx_as_percents:
            xx = xx * 100 / len(counts)
        y = counts
        return xx, y, lage, le
    positions = np.arange(len(counts))
    if len(counts) > 1:
        samples = np.repeat(positions, counts)
        # expand counts into sample positions
        kde: Callable = gaussian_kde(samples)
    else:
        kde = lambda x: np.full(x.shape, counts[0])
    if xx_as_percents:
        x = np.linspace(positions.min(), positions.max(), 400)
        y = kde(x)
        xx = x * 100 / len(counts)
    else:
        xx = np.arange(len(counts))
        y = kde(xx)
    return xx, y, lage, le


def quantify_normalized_video(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> None:
    """Quantify an already normalized video."""
    SG = ShapeGlobals()
    norm_video_path = get_normalized_video_fpath(video_path, config, metadata)
    if not norm_video_path.exists():
        raise RuntimeError(f"Normalized video {norm_video_path} does not exist.")
    fpath = get_le_lage_fpath(video_path, config, metadata)
    if fpath.exists() and not config.rewrite_if_exists:
        logger.info(
            f"Quantification file {fpath} already exists. Skipping quantification."
        )
        return
    lage, le = [], []
    for _, normalized_frame in generate_grayscale_frames(norm_video_path):
        normalized_frame[~SG.QUANT_MASK] = 0
        _quant, _lage, _le = _quantify_with_leading_lagging_edges(normalized_frame)
        lage.append(_lage)
        le.append(_le)
    _write_lagging_leading_edges(fpath, config, metadata, lage, le)
