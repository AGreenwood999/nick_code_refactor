import pathlib

import cv2
import numpy as np

"""
================= Flags =================
"""
REWRITE_IF_EXISTS = False

"""
================= Paths =================
"""
RUNS_TO_QUANTIFY = [
    "20251212 Runs",
    "20251210 Runs",
    "20251209 Runs",
    "20251203 Runs",
    "20251202 Runs",
]  # Subdirectory names to process automatically if no path is given to the main function.
# NOTE: ROOT_DIR is the directory that has subdirectories with videos to process.
ROOT_DIR = pathlib.Path(
    "G:\\Shared drives\\Theralode\\theralode-exp Drafts\\EXP_TBD_NMR IONP"
)
# NOTE: SAVE_DIR is the directory where output files will be saved (parent of the main script).
SAVE_DIR = pathlib.Path(__file__).parent.parent / "output"
NORM_VIDEOS_DIR = SAVE_DIR / "normalized_videos"
NORM_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PATH_SUFFIXES = [".wmv", ".mp4", ".avi"]
EMPTY_TUBE_WITH_MM_SCALE_VIDEO_PATH = (
    ROOT_DIR / "Empty Tube" / "Empty Tube 20251204.wmv"
)
EMPTY_TUBE_WITHOUT_MM_SCALE_VIDEO_PATH = (
    ROOT_DIR / "20251222 Blank Run" / "Settle___2025-12-22_11-45-18-2672-06-00.wmv"
)
VIDEO_METADATA_CACHE_PATH = pathlib.Path(__file__).parent.parent / "video_metadata.json"

"""
================= Times =================
"""
NUM_MINS_TO_PROCESS_IN_VIDEO: int = 5  # Limit to first 5 minutes
NUM_SECS_TO_PROCESS_IN_VIDEO: int = NUM_MINS_TO_PROCESS_IN_VIDEO * 60
TIME_OF_INTEREST: int = 120  # seconds

VIDEO_FRAMES_PER_REAL_TIME_SECONDS = {
    "2025-12-12": 3.7,
    "2025-12-10": 3.7,
    "2025-12-09": 3.7,  # 3.7 and 3.706 work, not 3.69 or 3.707 (these videos "skipped 7 frames" at presumably 30 fps capture)
    "2025-12-03": 1,
    "2025-12-02": 1,
}
DEFAULT_FRAMES_PER_REAL_TIME_SECONDS = (
    3.7  # This value will be used if the datestamp is not found in the above dict.
)

"""
================= Quantification =================
"""
BW_THRESHOLD = 45.0  # Threshold for binarizing the normalized frame.

"""
================= Distances =================
"""
MAGNET_DISTANCE_TO_MM = {0: 74, 1: 89, 2: 104, 3: 119, 4: 134, 5: 149, 6: 164}
PX_PER_MM = (
    541 - 246
) / 42.0  # As measured on the ruler in the Empty Tube (with mm scale) video.


class ShapeGlobals:
    """
    Just so we don't forget where these numbers came from, here are the main measurements in csv format:
        trial_name,tube_tl_x,tube_tl_y,tube_bl_x,tube_bl_y,tube_tr_x,tube_tr_y,tube_br_x,tube_br_y
        empty_tube,111.0,325.0,111.0,362.0,1280.0,348.0,1280.0,384.0
    """

    def __init__(self, has_mm_scale: bool):
        self.has_mm_scale = has_mm_scale

        shift_down_left_side = (
            0 if has_mm_scale else 6
        )  # Because the tube is a few pixels higher in "Empty Tube without mm scale" video.
        shift_down_right_side = (
            0 if has_mm_scale else 9
        )  # Because the tube in "Empty Tube without mm scale" video is slightly tilted.

        self.TUBE_TL = [
            111.0,
            325.0 - shift_down_left_side,
        ]  # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_BL = [
            111.0,
            362.0 - shift_down_left_side,
        ]  # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_TR = [
            1280.0,
            348.0 - shift_down_right_side,
        ]  # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_BR = [
            1280.0,
            384.0 - shift_down_right_side,
        ]  # Measured in the Empty Tube (with mm scale) video.

        self.TUBE_TR_POST_TRANSFORM = [self.TUBE_TR[0], self.TUBE_TL[1]]
        self.TUBE_BR_POST_TRANSFORM = [self.TUBE_BR[0], self.TUBE_BL[1]]

        self.VIAL_DIAMETER_PX = (
            384 - 347
        )  # Distance between the edges of the vial in the Empty Tube (with mm scale) video.
        self.CORNERS_DIST_PX = (
            367 - 316
        )  # Distance between the corners of the vial fingers in the Empty Tube (with mm scale) video.

        """
        ================= Alignment =================
        """
        self.VIAL_LEFTMOST_PX = 75  # Position of leftmost pixel of the vial as measured in the Empty Tube (with mm scale) video.
        _PTS_SRC = np.asarray(
            [self.TUBE_TL, self.TUBE_TR, self.TUBE_BR, self.TUBE_BL], dtype=np.float32
        )
        _PTS_DST = np.asarray(
            [
                self.TUBE_TL,
                self.TUBE_TR_POST_TRANSFORM,
                self.TUBE_BR_POST_TRANSFORM,
                self.TUBE_BL,
            ],
            dtype=np.float32,
        )
        self.STRAIGHTEN_M = cv2.getPerspectiveTransform(_PTS_SRC, _PTS_DST)
        self.POST_STRAIGHTEN_TUBE_TL = (self.VIAL_LEFTMOST_PX, self.TUBE_TL[1])
        self.POST_STRAIGHTEN_TUBE_BL = (self.VIAL_LEFTMOST_PX, self.TUBE_BL[1])

        self.FINAL_FRAME_SHAPE: np.ndarray = np.asarray(
            [
                int(self.TUBE_BR_POST_TRANSFORM[1])
                - int(self.POST_STRAIGHTEN_TUBE_TL[1]),
                int(self.TUBE_BR_POST_TRANSFORM[0])
                - int(self.POST_STRAIGHTEN_TUBE_TL[0]),
            ]
        )
        # SHAPE_ADDEND trims off the top and left to ensure even dimensions for ffmpeg
        self.SHAPE_ADDEND = [
            -(self.FINAL_FRAME_SHAPE[0] % 2),
            -(self.FINAL_FRAME_SHAPE[1] % 2),
        ]
        self.FINAL_FRAME_SHAPE = (
            self.FINAL_FRAME_SHAPE + self.SHAPE_ADDEND
        )  # Ensure even dimensions for ffmpeg

        """
        ================= Masking =================
        """
        top_trim_px = 3
        bottom_trim_px = 14 if has_mm_scale else 3
        left_trim_px = 35
        right_trim_px = 150
        conservative_left_right_trim_px = 200
        self.QUANT_MASK = np.zeros(self.FINAL_FRAME_SHAPE, dtype=np.bool_)
        self.CONSERVATIVE_QUANT_MASK = self.QUANT_MASK.copy()
        self.QUANT_MASK[
            top_trim_px
            + self.SHAPE_ADDEND[
                0
            ] : -bottom_trim_px,  # Amount to remove from top and bottom
            left_trim_px
            + self.SHAPE_ADDEND[
                1
            ] : -right_trim_px,  # Amount to remove from left and right
        ] = 1

        # Trim more from the left and right sides for conservative quantification
        self.CONSERVATIVE_QUANT_MASK[
            top_trim_px + self.SHAPE_ADDEND[0] : -bottom_trim_px,
            conservative_left_right_trim_px
            + self.SHAPE_ADDEND[1] : -conservative_left_right_trim_px,
        ] = 1
