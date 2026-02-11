import logging
import pathlib

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    VideoContext,
)
from nmr_particle_motion.frame_generator import (
    VideoData,
    generate_grayscale_frames,
)
from nmr_particle_motion.shapeglobals import ShapeGlobals
from nmr_particle_motion.video_writer import VideoWriter

logger = logging.getLogger("nmr_particle_motion")


def align_images_phase_correlation(reference_image, image_to_align):
    # Convert to float32 grayscale
    def to_float_gray(img):
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        return img.astype(np.float32)

    ref = to_float_gray(reference_image)
    img = to_float_gray(image_to_align)

    # Ensure same size (crop or pad if needed)
    h, w = min(ref.shape[0], img.shape[0]), min(ref.shape[1], img.shape[1])
    ref = ref[:h, :w]
    img = img[:h, :w]

    # Phase correlation gives (y, x) shift
    shift, _ = cv2.phaseCorrelate(img, ref)
    dx, dy = shift[0], shift[1]
    return dx, dy


def clip_frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Clip the frame values to uint8."""
    return np.clip(frame, 0, 255).astype(np.uint8)


def plot_frame(frame: np.ndarray, title: str = "Frame", ax=None):
    """Plot a single frame using matplotlib."""
    if ax is None:
        _fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(clip_frame_to_uint8(frame), cv2.COLOR_BGR2RGB), cmap="gray")
    ax.set_title(title)
    ax.set_axis_off()


class VideoNormalizer:
    """Class to normalize video frames."""

    null_frame: np.ndarray | None = None

    def __init__(self, ctx: VideoContext, config: Config):
        # Just to assert it's a pre-normalized video.
        self.video_path = ctx.path
        self.norm_path = ctx.norm_path

        self.SG = ShapeGlobals()

        if VideoNormalizer.null_frame is None:
            VideoNormalizer.null_frame = self.get_null_frame(config)

        self.null_frame = VideoNormalizer.null_frame

        self.cropped_null_frame = self.crop_frame_to_tube(self.null_frame)
        self.bw_threshold = config.bw_threshold_default
        self.binarization_margin = 0
        self.brightness_scale_factor = 1.0

    """ Utilities for normalizing video frames.
    """

    def straighten_image(self, aligned_image: np.ndarray) -> np.ndarray:
        """Straighten the image using perspective transform.
        The image must be aligned with the null frame before straightening.
        """
        H, W = aligned_image.shape[:2]
        warped = cv2.warpPerspective(aligned_image, self.SG.STRAIGHTEN_M, (W, H))
        return warped

    def crop_to_tube_rectangle(
        self,
        frame: np.ndarray,
        was_straightened: bool,
        margin: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> np.ndarray:
        """Get the tube rectangle from the frame based on fixed coordinates.
        Margin: (top, bottom, left, right)
        """
        tl_x, tl_y = int(self.SG.TUBE_TL[0]), int(self.SG.TUBE_TL[1])
        br_x, br_y = int(self.SG.TUBE_BR[0]), int(self.SG.TUBE_BR[1])
        if was_straightened:
            tl_x, tl_y = (
                int(self.SG.POST_STRAIGHTEN_TUBE_TL[0]),
                int(self.SG.POST_STRAIGHTEN_TUBE_TL[1]),
            )
            br_x, br_y = (
                int(self.SG.TUBE_BR_POST_TRANSFORM[0]),
                int(self.SG.TUBE_BR_POST_TRANSFORM[1]),
            )
        return frame[
            tl_y - self.SG.SHAPE_ADDEND[0] - margin[0] : br_y + margin[1],
            tl_x - self.SG.SHAPE_ADDEND[1] - margin[2] : br_x + margin[3],
        ]

    def crop_to_final_shape(self, straightened_frame: np.ndarray) -> np.ndarray:
        """Crop the frame to the final shape used for quantification."""
        f = self.crop_to_tube_rectangle(straightened_frame, was_straightened=True)
        assert f.shape == tuple(self.SG.FINAL_FRAME_SHAPE), (
            f"Final cropped frame shape {f.shape} does not match expected {self.SG.FINAL_FRAME_SHAPE}"
        )
        return f

    def show_roi_sanity_check(
        self,
        frame: np.ndarray,
        was_straightened: bool,
        margin: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """Displays the reference square and tube rectangle for visual confirmation."""
        plot_frame(self.crop_to_tube_rectangle(frame, was_straightened, margin=margin))

    @staticmethod
    def get_null_frame(config: Config) -> np.ndarray:
        """Get the null frame (empty tube) for normalization."""
        null_vpath = config.empty_tube_video_path
        vd = VideoData.from_path(null_vpath)
        vd.video.release()  # Release immediately since we only need metadata here.
        frames_to_capture = np.linspace(
            0, vd.nframes - 1, num=min(50, vd.nframes), dtype=int
        )
        frames = []
        for i, frame in enumerate(generate_grayscale_frames(null_vpath)):
            if i in frames_to_capture:
                frames.append(frame)

        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def subtract_null_frame(self, A: np.ndarray) -> np.ndarray:
        """Subtracts either the full or cropped null frame from A
        (depending on image shape). Accepts float or uint8 arrays.
        Result will be uint8.

        Subtract frames such that result is A - B.
        This highlights differences between A and B.
        B is a frame showing an empty vial,
        and A is the current frame being processed.

        Example
        -------
        subtracted_frame = subtract_frame(frame, first_frame)
        """
        if not np.issubdtype(A.dtype, np.floating):
            A = A.astype(np.float32)
        ref = self.null_frame
        if A.shape == self.cropped_null_frame.shape:
            ref = self.cropped_null_frame
        return np.abs(A - ref).astype(np.uint8)

    def align_to_null_frame(self, frame: np.ndarray) -> np.ndarray:
        """Align the frame by centering it around the center of mass.
        This is useful for ensuring consistent alignment across frames.
        """
        margin = (50, 50, 50, 50)
        r0 = self.crop_to_tube_rectangle(
            self.null_frame,  # type:ignore
            was_straightened=False,
            margin=margin,
        )
        r1 = self.crop_to_tube_rectangle(frame, was_straightened=False, margin=margin)
        shift_x, shift_y = align_images_phase_correlation(r0, r1)
        if np.allclose([shift_x, shift_y], [0, 0], atol=1e-2):
            return frame
        translation_matrix = np.asarray(
            [[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32
        )
        aligned_frame = cv2.warpAffine(frame, translation_matrix, frame.shape[:2][::-1])
        return aligned_frame

    def crop_frame_to_tube(self, frame: np.ndarray) -> np.ndarray:
        """Aligns to the null frame, straightens, and crops to the tube rectangle."""
        return self.crop_to_final_shape(
            self.straighten_image(self.align_to_null_frame(frame))
        )

    def tune_brightness_scale_factor(self, frame: np.ndarray):
        """Tune the brightness scale factor to match the null frame
        using median pixel values within the quantification mask.
        """
        cropped_frame = self.crop_frame_to_tube(frame)
        tuning_mask = self.SG.CONSERVATIVE_QUANT_MASK.copy()
        mean_null = np.median(self.cropped_null_frame[tuning_mask])
        mean_frame = np.median(cropped_frame[tuning_mask])
        if mean_frame != 0:
            self.brightness_scale_factor = mean_null / mean_frame

    def scale_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Scale the brightness of the frame to match the null frame.
        This helps to normalize lighting conditions across frames.
        """
        if not np.issubdtype(frame.dtype, np.floating):
            frame = frame.astype(np.float32)
        return frame * self.brightness_scale_factor

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize the frame by subtracting the null frame and aligning it.
        This prepares the frame for further analysis.
        """
        if frame.shape != self.null_frame.shape:  # type:ignore
            # For now, assume size mismatch is an error. Otherwise, resize.
            # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            raise ValueError("Frame shape does not match null frame shape.")
        cropped_frame = self.crop_frame_to_tube(frame)
        blurred = uniform_filter(cropped_frame, size=3)
        scaled = self.scale_brightness(blurred)
        _, thresholded_frame = cv2.threshold(
            self.subtract_null_frame(scaled), self.bw_threshold, 255, cv2.THRESH_BINARY
        )
        # thresholded_frame = (scaled < (self.cropped_null_frame.astype(np.float32)-self.binarization_margin)) * 255 # This method makes strange-looking videos. The cloud is a bunch of evenly-spaced dots.
        return thresholded_frame

    def tune_binarization_parameters(
        self, frame: np.ndarray, config: Config, show=False
    ) -> float:
        """Tunes brightness scale factor and binarization margin using the provided frame."""
        fig = None
        axs = None
        if show:
            fig, axs = plt.subplots(4, 1, figsize=(12, 6))
            cropped = self.crop_frame_to_tube(frame)
            blurred = uniform_filter(cropped, size=3)
            scaled = self.scale_brightness(blurred)
            thresholded_frame = (
                scaled < (self.cropped_null_frame - self.binarization_margin)
            ) * 255
            plot_frame(scaled, title="Scaled Frame for Tuning", ax=axs[0])
            plot_frame(
                self.cropped_null_frame, title="Null Frame for Tuning", ax=axs[1]
            )
            plot_frame(
                thresholded_frame, title="Possible binary mask before tuning", ax=axs[2]
            )
        conservative_quant_mask = self.SG.CONSERVATIVE_QUANT_MASK.copy()
        self.binarization_margin = 0
        self.bw_threshold = config.bw_threshold_default
        f = self.normalize_frame(frame.copy())
        while (
            f[conservative_quant_mask].sum() != 0
            and self.binarization_margin < 255
            and self.bw_threshold < 255
        ):
            self.binarization_margin += 1
            self.bw_threshold += 1
            f = self.normalize_frame(frame.copy())
        if show and fig is not None and axs is not None:
            f[~conservative_quant_mask] = 0
            plot_frame(
                f,
                title=f"Final binary mask after tuning (should look black, BW threshold={self.bw_threshold})",
                ax=axs[3],
            )
            fig.tight_layout()
            plt.show()
        print(f"Tuned BW threshold to {self.binarization_margin}")
        return self.binarization_margin

    def normalize_video(self, config: Config) -> None:
        """Perform the quantification of bead mixing in the video."""
        if self.norm_path.exists() and not config.overwrite_norm_vid:
            logger.info(
                f"Normalized video {self.norm_path} already exists. Skipping normalization."
            )
            return
        is_first = True
        with VideoWriter(
            outpath=self.norm_path,
            fps=30,
            width=self.SG.FINAL_FRAME_SHAPE[1],
            height=self.SG.FINAL_FRAME_SHAPE[0],
            overwrite=True,
            save_all_data=True,
        ) as video_writer:
            for frame in tqdm(generate_grayscale_frames(self.video_path)):
                if is_first:
                    self.tune_brightness_scale_factor(frame)
                    # self.tune_binarization_parameters(frame)
                    is_first = False
                normalized_frame = self.normalize_frame(frame)
                quant_frame = normalized_frame.copy()
                quant_frame[~self.SG.QUANT_MASK] = 0
                video_writer.write_frame(quant_frame)
