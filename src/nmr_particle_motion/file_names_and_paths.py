"""
File and path utilities for particle analysis.
"""

import matplotlib.pyplot as plt
import logging
import polars as pl
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger("nmr_particle_motion")


@dataclass(frozen=True)
class RunDetails:
    substance: str
    lot: str
    vial_number: int
    mass: float
    name: str

    @classmethod
    def from_dict(cls, d: dict, name: str) -> "RunDetails":
        return cls(
            substance=d.get("substance", ""),
            lot=d.get("lot", "Unknown lot"),
            vial_number=d.get("vial number", -1),
            mass=d.get("mass", -1.0),
            name=name,
        )


@dataclass(frozen=True)
class VideoContext:
    path: Path
    norm_path: Path
    details: RunDetails
    rotation_matrix: np.ndarray
    normalization_theshold: int


def get_video_details_from_path(path: Path) -> RunDetails:
    csv_path = path.parent / "run_details.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"Run details CSV not found at {csv_path}")

    all_run_details = pl.read_csv(csv_path)
    return RunDetails.from_dict(
        all_run_details.filter(pl.col("name") == path.name).to_dict(), path.stem
    )


def get_first_frame(vid_path: Path, gray: bool = True):
    vid_cap = cv2.VideoCapture(vid_path)
    ret, img = vid_cap.read()
    vid_cap.release()

    if not ret:
        raise RuntimeError(f"Failed to get first frame of video {vid_path.stem}")

    if gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def get_null_frame(ctx: VideoContext) -> np.ndarray:
    null_frame = get_first_frame(ctx.path, gray=True)
    return cv2.warpAffine(null_frame, ctx.rotation_matrix, null_frame.shape[::-1])


def _get_tube_lines_backup(canny: np.ndarray, roi_left: int, roi_right: int):
    p1_where = np.argwhere(canny[:, roi_left]).flatten()
    p2_where = np.argwhere(canny[:, roi_right]).flatten()

    if len(p1_where) != 4 or len(p2_where) != 4:
        return np.array([])

    lines = np.empty((2, 1, 4), dtype=np.int_)
    lines[0, 0, :] = [roi_left, p1_where[1], roi_right, p2_where[1]]
    lines[1, 0, :] = [roi_left, p1_where[2], roi_right, p2_where[2]]
    return lines


def get_tube_lines(
    img: np.ndarray,
    *,
    roi_left=200,
    roi_right=1000,
    min_line_length=100,
    max_line_gap=50,
    hough_threshold=50,
    canny_low=100,
    canny_high=200,
) -> np.ndarray:
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur_img, canny_low, canny_high)
    roi_top = np.argwhere(canny[:, roi_left])[0][0] + 20
    roi_bottom = np.argwhere(canny[:, roi_right])[-1][0] - 20

    lines = cv2.HoughLinesP(
        canny[roi_top:roi_bottom, roi_left:roi_right],
        0.5,
        np.pi / 360,
        hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        print("HAD TO DO BACKUP")
        return _get_tube_lines_backup(canny, roi_left, roi_right)

    for i in range(len(lines)):
        lines[i, 0, 0] += roi_left
        lines[i, 0, 1] += roi_top
        lines[i, 0, 2] += roi_left
        lines[i, 0, 3] += roi_top

    return lines


def get_left_boundary(
    img: np.ndarray,
) -> int:
    sobel = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    abs_sobel_mean = np.mean(np.absolute(sobel), axis=0)
    abs_sobel_mean_smooth = gaussian_filter1d(abs_sobel_mean, sigma=1.5)
    peak, _ = find_peaks(
        abs_sobel_mean_smooth,
        prominence=50,
        distance=400,
    )
    if len(peak) == 0:
        raise RuntimeError("Unable to find left edge of tube.")

    return peak[0]


def get_vertical_boundaries(
    img: np.ndarray, *, roi_left: int = 199, roi_right: int = 1000
) -> np.ndarray:
    sobel = cv2.Sobel(img, cv2.CV_32F, 0, 1)[:, roi_left:roi_right]
    abs_sobel_mean = np.mean(np.absolute(sobel), axis=1)
    abs_sobel_mean_smooth = gaussian_filter1d(abs_sobel_mean, sigma=1.5)
    peaks, _ = find_peaks(
        abs_sobel_mean_smooth,
        prominence=20,
        distance=20,  # enforce separation in pixels
    )

    if len(peaks) != 4:
        raise RuntimeError(
            "Failed to find 4 expected boundaries: bottom and top of fixture, bottom and top of tube."
        )

    return np.asarray(peaks)


def get_analysis_boundaries(
    ctx: VideoContext,
    *,
    test_roi_left: int = 199,
    test_roi_right: int = 1000,
    margin: int = 5,
) -> tuple[int, int, int]:
    img = get_null_frame(ctx)
    _, top, bottom, _ = get_vertical_boundaries(
        img, roi_left=test_roi_left, roi_right=test_roi_right
    )
    left = get_left_boundary(img[top:bottom, :])

    return top + margin, bottom - margin, left + margin


def get_rotation_matrix_to_align_tube(
    vid_path: Path,
    *,
    roi_left=200,
    roi_right=1000,
    min_line_length=500,
    max_line_gap=100,
    hough_threshold=50,
    canny_low=100,
    canny_high=200,
) -> np.ndarray:
    img = get_first_frame(vid_path, gray=True)
    lines = get_tube_lines(
        img,
        roi_left=roi_left,
        roi_right=roi_right,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        hough_threshold=hough_threshold,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    if lines.size == 0:
        raise RuntimeError("Failed to straighten tube")

    fits = np.empty((len(lines),), dtype=np.float32)
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            fits[i] = np.polyfit([x1, x2], [y1, y2], 1)[0]

    h, w = img.shape
    center = (w / 2, h / 2)
    angle = np.rad2deg(np.arctan(np.mean(fits)))

    return cv2.getRotationMatrix2D(center, angle, 1)


def get_all_video_contexts_from_directory(videos_dir: Path) -> list[VideoContext]:
    contexts = []
    for video_path in videos_dir.glob("*.wmv", case_sensitive=False):
        if not video_path.is_file() or video_path.stem.endswith("_normalized"):
            continue

        norm_video_path = video_path.with_name(
            f"{video_path.stem}_normalized{video_path.suffix}"
        )

        contexts.append(
            VideoContext(
                video_path.expanduser().resolve(),
                norm_video_path.expanduser().resolve(),
                get_video_details_from_path(video_path),
                get_rotation_matrix_to_align_tube(video_path),
                -50,
            )
        )

    return contexts
