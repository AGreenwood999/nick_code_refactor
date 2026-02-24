"""Quantify bead mixing in a video."""

import logging
from pathlib import Path

import cv2
import numpy as np
import typer
from matplotlib import pyplot as plt
from nmr_particle_motion_new.file_names_and_paths import (
    VideoContext,
    get_all_video_contexts_from_directory,
    get_analysis_boundaries,
    get_null_frame,
)

logger = logging.getLogger("nmr_particle_motion")


APP = typer.Typer()


def get_video_frames(ctx: VideoContext, gray: bool = True):
    vid_cap = cv2.VideoCapture(ctx.path)

    while True:
        ret, img = vid_cap.read()

        if not ret:
            break

        if gray is True:
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            yield img

    vid_cap.release()


def normalize_frame(
    frame: np.ndarray,
    ctx: VideoContext,
    boundaries: tuple[int | None, int | None, int | None],
    null_frame_cropped: np.ndarray | None = None,
    *,
    right_cut_off: int | None = None,
    gaussian_kernel: tuple[int, int] = (3, 3),
    gaussian_sigma: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    top_boundary, bottom_boundary, left_boundary = boundaries

    if null_frame_cropped is None:
        null_frame = get_null_frame(ctx)
        null_frame_blurr = cv2.GaussianBlur(null_frame, gaussian_kernel, gaussian_sigma)
        null_frame_cropped = null_frame_blurr[
            top_boundary:bottom_boundary, left_boundary:right_cut_off
        ].astype(np.float32)

    straight_frame = cv2.warpAffine(frame, ctx.rotation_matrix, frame.shape[::-1])
    straight_frame_blurred = cv2.GaussianBlur(
        straight_frame, gaussian_kernel, gaussian_sigma
    )
    straight_frame_cropped = straight_frame_blurred[
        top_boundary:bottom_boundary, left_boundary:right_cut_off
    ].astype(np.float32)

    normalized = straight_frame_cropped - null_frame_cropped
    normalized[normalized > ctx.normalization_theshold] = 0
    return normalized, null_frame_cropped


def normalize_video(
    ctx: VideoContext,
    *,
    right_cut_off: int | None = None,
    writer_fourcc: int = 1983148141,
    writer_fps: int = 30,
):
    top, bottom, left = get_analysis_boundaries(ctx)
    null_frame_cropped = get_null_frame(ctx)[top:bottom, left:right_cut_off]

    vid_writer = cv2.VideoWriter(
        ctx.norm_path,
        writer_fourcc,
        writer_fps,
        null_frame_cropped.shape[::-1],
        isColor=False,
    )

    for i, frame in enumerate(get_video_frames(ctx)):
        normalized, _ = normalize_frame(
            frame,
            ctx,
            (top, bottom, left),
            null_frame_cropped,
            right_cut_off=right_cut_off,
        )

        vid_writer.write(normalized.astype(np.uint8))

    vid_writer.release()


@APP.command()
def testing(videos_dir: Path = Path("data/testing")):
    contexts = get_all_video_contexts_from_directory(videos_dir)

    for ctx in contexts:
        normalize_video(ctx, right_cut_off=1000)


if __name__ == "__main__":
    typer.run(testing)
