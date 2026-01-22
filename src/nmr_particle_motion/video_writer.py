import pathlib
import subprocess

import numpy as np
import logging

logger = logging.getLogger("nmr_particle_motion")


class VideoWriter:
    """Class to handle video writing using ffmpeg.
    This class provides a method to write a sequence of frames to a video file.
    Frames must be grayscale and type uint8.
    """

    outpath: pathlib.Path
    fps: int
    width: int
    height: int
    process: subprocess.Popen | None
    can_write: bool

    def __init__(
        self,
        outpath: pathlib.Path,
        fps: int,
        width: int,
        height: int,
        overwrite: bool = True,
        save_all_data: bool = True,
    ):
        self.outpath = pathlib.Path(outpath)
        self.fps = fps
        if width % 2 or height % 2:
            raise ValueError(
                "Width and height must be even numbers for ffmpeg compatibility."
            )
        self.width = width
        self.height = height
        self.process = None
        self.can_write = ((not outpath.exists()) or overwrite) and save_all_data
        if outpath.exists() and overwrite:
            outpath.unlink()

    def __enter__(self):
        """Context manager entry method."""
        self.start_writing()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit method."""
        self.close_and_wait()
        return False

    def start_writing(self):
        """Open streams so the video can be written."""
        if not self.can_write:
            print("Video exists and overwrite is set to False. Not saving video.")
            return
        logger.info(f"Saving video to {self.outpath}...")
        # Start ffmpeg subprocess
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "y8",
            "-r",
            str(self.fps),
            "-i",
            "-",  # read from stdin
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            self.outpath.as_posix(),
        ]
        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to the video file.
        The frame must be type uint8
        (and probably needs to be contiguous,
        so consider that if you see a broken pipe error.)
        """
        if (
            self.can_write
            and self.process is not None
            and self.process.stdin is not None
        ):
            self.process.stdin.write(frame.tobytes())

    def close_and_wait(self) -> bool:
        """Close the video writer and wait for the process to finish."""
        if (
            self.can_write
            and self.process is not None
            and self.process.stdin is not None
        ):
            self.process.stdin.close()
            self.process.wait()
            logger.info(f"Video saved as {self.outpath}")
        return self.can_write
