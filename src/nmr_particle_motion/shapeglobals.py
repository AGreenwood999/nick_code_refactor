import cv2
import numpy as np

NUM_MINS_TO_PROCESS_IN_VIDEO: int = 5  # Limit to first 5 minutes
NUM_SECS_TO_PROCESS_IN_VIDEO: int = NUM_MINS_TO_PROCESS_IN_VIDEO * 60


class ShapeGlobals:
    """
    Just so we don't forget where these numbers came from, here are the main measurements in csv format:
        trial_name,tube_tl_x,tube_tl_y,tube_bl_x,tube_bl_y,tube_tr_x,tube_tr_y,tube_br_x,tube_br_y
        empty_tube,111.0,325.0,111.0,362.0,1280.0,348.0,1280.0,384.0
    """

    def __init__(self):
        # Because the tube is a few pixels higher in "Empty Tube without mm scale" video.
        shift_down_left_side = 0
        # Because the tube in "Empty Tube without mm scale" video is slightly tilted.
        shift_down_right_side = 0
        # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_TL = [
            111.0,
            325.0 - shift_down_left_side,
        ]
        # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_BL = [
            111.0,
            362.0 - shift_down_left_side,
        ]
        # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_TR = [
            1280.0,
            348.0 - shift_down_right_side,
        ]
        # Measured in the Empty Tube (with mm scale) video.
        self.TUBE_BR = [
            1280.0,
            384.0 - shift_down_right_side,
        ]

        self.TUBE_TR_POST_TRANSFORM = [self.TUBE_TR[0], self.TUBE_TL[1]]
        self.TUBE_BR_POST_TRANSFORM = [self.TUBE_BR[0], self.TUBE_BL[1]]

        # Distance between the edges of the vial in the Empty Tube (with mm scale) video.
        self.VIAL_DIAMETER_PX = 384 - 347
        # Distance between the corners of the vial fingers in the Empty Tube (with mm scale) video.
        self.CORNERS_DIST_PX = 367 - 316

        """
        ================= Alignment =================
        """
        # Position of leftmost pixel of the vial as measured in the Empty Tube (with mm scale) video.
        self.VIAL_LEFTMOST_PX = 75
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
        # Ensure even dimensions for ffmpeg
        self.FINAL_FRAME_SHAPE = self.FINAL_FRAME_SHAPE + self.SHAPE_ADDEND

        """
        ================= Masking =================
        """
        top_trim_px = 3
        bottom_trim_px = 14
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
