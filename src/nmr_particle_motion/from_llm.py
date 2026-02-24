"""
Skeleton analysis pipeline for particle motion quantification.
This file intentionally contains NO filesystem logic inside core functions.
Fill in implementations incrementally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# =============================
# Data models
# =============================


@dataclass(frozen=True)
class AnalysisParams:
    bw_threshold: float
    frames_per_second: float
    mm_per_pixel: float
    valid_mask: np.ndarray


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    column_counts: np.ndarray
    lagging_edge_px: int
    leading_edge_px: int


@dataclass(frozen=True)
class VideoResult:
    frames: list[FrameResult]


# =============================
# Core scientific functions
# =============================


def normalize_frame(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Normalize a frame using a reference image."""
    raise NotImplementedError


def segment_particles(frame: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a grayscale frame to a binary particle mask."""
    raise NotImplementedError


def count_particles_per_column(
    binary_frame: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Count particle pixels per column."""
    raise NotImplementedError


def find_edges(column_counts: np.ndarray) -> tuple[int, int]:
    """Find lagging and leading edges from column counts."""
    raise NotImplementedError


def extract_particle_distribution(
    column_counts: np.ndarray, lagging: int, leading: int
) -> np.ndarray:
    """Extract particle distribution between edges."""
    raise NotImplementedError


# =============================
# Composition functions
# =============================


def quantify_frame(
    frame_index: int,
    frame: np.ndarray,
    reference: np.ndarray,
    params: AnalysisParams,
) -> FrameResult:
    """Quantify a single frame."""
    norm = normalize_frame(frame, reference)
    binary = segment_particles(norm, params.bw_threshold)
    counts = count_particles_per_column(binary, params.valid_mask)
    lag, lead = find_edges(counts)

    return FrameResult(
        frame_index=frame_index,
        column_counts=counts,
        lagging_edge_px=lag,
        leading_edge_px=lead,
    )


def quantify_video(
    frames: Iterable[np.ndarray],
    reference: np.ndarray,
    params: AnalysisParams,
) -> VideoResult:
    """Quantify all frames in a video."""
    results: list[FrameResult] = []

    for i, frame in enumerate(frames):
        results.append(
            quantify_frame(
                frame_index=i,
                frame=frame,
                reference=reference,
                params=params,
            )
        )

    return VideoResult(frames=results)


# =============================
# Unit conversion helpers
# =============================


def frame_to_time(frame_index: int, fps: float) -> float:
    """Convert frame index to seconds."""
    return frame_index / fps


def pixel_to_distance(pixel_index: int, mm_per_pixel: float) -> float:
    """Convert pixel index to millimeters."""
    return pixel_index * mm_per_pixel


# =============================
# I/O boundary (thin wrappers)
# =============================


def load_video_frames(path: Path) -> Iterable[np.ndarray]:
    """Yield frames from a video file."""
    raise NotImplementedError


def load_reference_image(path: Path) -> np.ndarray:
    """Load reference normalization image."""
    raise NotImplementedError


def write_edges_csv(result: VideoResult, path: Path, params: AnalysisParams) -> None:
    """Write edge results to CSV."""
    raise NotImplementedError


# =============================
# Top-level pipeline
# =============================


def analyze_video(
    video_path: Path,
    reference_path: Path,
    params: AnalysisParams,
) -> VideoResult:
    """Run full analysis pipeline on a single video."""
    frames = load_video_frames(video_path)
    reference = load_reference_image(reference_path)
    return quantify_video(frames, reference, params)
