# Combined Python codebase

Total files: 12
Base directory: /home/augustus/Documents/nmr_particle_motion/src/nmr_particle_motion

## FILE: __init__.py

```python

```

## FILE: cli.py

```python
"""Quantify bead mixing in a video."""

import json
import logging
import pathlib
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    generate_video_paths,
    get_coated_vs_uncoated_plot_path,
    get_dist_at_time_plot_fpath,
    get_full_video_distributions_fpaths,
    get_height_by_dist_at_time_fpath,
    get_height_over_time_fpath,
    get_le_lage_fpath,
    video_path_to_run_details,
)
from nmr_particle_motion.leading_lagging_edge_and_particle_dist import (
    quantify_normalized_video,
)
from nmr_particle_motion.plot_output import (
    plot_distribution,
    plot_distribution_by_time,
    plot_height_at_time_of_interest_by_distance,
    plot_height_over_time,
    plot_video_as_image,
)
from nmr_particle_motion.video_normalizing import VideoNormalizer

logger = logging.getLogger("nmr_particle_motion")


def get_substance_mass_rpm_combos(path_to_videos: str | pathlib.Path, config: Config):
    substance_mass_rpm_combos = set()
    for _, vdesc, _ in generate_video_paths(path_to_videos, config):
        substance_mass_rpm_combos.add((vdesc.substance, vdesc.mass, vdesc.rpm))

    return substance_mass_rpm_combos


def maximize_figure() -> None:
    """Set the size of a fullscreen figure."""
    mgr = plt.get_current_fig_manager()
    if mgr is not None:
        mgr.full_screen_toggle()


def save_distribution_plot(
    video_path: pathlib.Path,
    config: Config,
    metadata: dict[str, str],
    time_sec: int,
    show: bool,
) -> None:
    """Plot the distribution of particles at a given time.
    This method visualizes the quantification results.
    """
    plot_fpath = get_dist_at_time_plot_fpath(video_path, config, metadata, time_sec)
    if plot_fpath.exists() and not config.rewrite_if_exists:
        logger.info(
            f"Distribution file {plot_fpath} already exists. Skipping distribution plotting."
        )
        return
    fig, ax = plt.subplots()
    plot_distribution(video_path, config, metadata, time_sec, ax)
    fig.tight_layout()
    plt.savefig(plot_fpath)
    if show:
        plt.show()
    plt.close(fig)


def save_distribution_by_time_plot(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], show: bool
) -> None:
    """Plot the distribution of particles at a given time.
    This method visualizes the quantification results.
    """
    fpath, plot_fpath = get_full_video_distributions_fpaths(
        video_path, config, metadata
    )
    if fpath.exists() and plot_fpath.exists() and not config.rewrite_if_exists:
        logger.info(
            f"Distribution file {fpath} already exists. Skipping distribution plotting."
        )
        return
    fig, ax = plt.subplots()
    distributions = plot_distribution_by_time(video_path, config, metadata, ax)
    fig.tight_layout()
    plt.savefig(plot_fpath)
    if show:
        plt.show()
    plt.close(fig)
    with open(fpath, "wt", encoding="utf-8") as ofh:
        pd.DataFrame(distributions).to_csv(
            ofh, header=False, index=False, lineterminator="\n"
        )


def normalize_videos(
    path_to_videos: str | pathlib.Path,
    config: Config,
    metadata: dict[str, str],
) -> None:
    for vpath, vdesc, _ in generate_video_paths(path_to_videos, config):
        logger.info(f"Normalizing video at time {vdesc.datestamp}")
        VideoNormalizer(vpath, config, metadata).normalize_video(config, metadata)


def quantify_normalized_videos(
    path_to_videos: str | pathlib.Path, config: Config, metadata: dict[str, str]
) -> None:
    for vpath, vdesc, _ in tqdm(generate_video_paths(path_to_videos, config)):
        logging.info(f"Quantifying run at time {vdesc.datestamp}")
        quantify_normalized_video(vpath, config, metadata)
        save_distribution_plot(
            vpath, config, metadata, time_sec=config.time_of_interest_sec, show=False
        )
        save_distribution_by_time_plot(vpath, config, metadata, show=False)


def plot_height_over_time_all_videos(
    path_to_videos: str | pathlib.Path, config: Config, metadata: dict[str, str]
):
    for vpath, _, _ in generate_video_paths(path_to_videos, config):
        quant_path = get_le_lage_fpath(vpath, config, metadata)
        fig, ax = plt.subplots()
        plot_height_over_time(quant_path, config, metadata, ax)
        ax.legend()
        fig.tight_layout()
        plt.savefig(
            get_height_over_time_fpath(vpath, config, metadata).with_suffix(".png")
        )
        plt.close(fig)


def plot_all_height_over_time_all_videos(
    path_to_videos: str | pathlib.Path, config: Config, metadata: dict[str, str]
):
    smrs = sorted(list(get_substance_mass_rpm_combos(path_to_videos, config)))
    fig, axs = plt.subplots(int(np.ceil(len(smrs) / 3)), 3, sharex=True, sharey=True)
    axs = axs.flatten()
    for vpath, vd, _ in generate_video_paths(path_to_videos, config):
        quant_path = get_le_lage_fpath(vpath, config, metadata)
        i = smrs.index((vd.substance, vd.mass, vd.rpm))
        plot_height_over_time(
            quant_path, config, metadata, axs[i], title_includes_magnet_distance=False
        )
    fig.suptitle("Particle Height Over Time")
    maximize_figure()
    fig.tight_layout()
    grandparent = get_height_over_time_fpath(vpath, config, metadata).parent.parent  # type:ignore
    figpath = grandparent / "all_height_over_time.png"
    if figpath.exists():
        logging.warning(f"{figpath} will be overwritten.")
    plt.savefig(figpath)
    plt.close(fig)


def plot_height_at_time_of_interest_by_distance_all_videos(
    path_to_videos: str | pathlib.Path, config: Config, metadata: dict[str, str]
):
    """Plot height at TIME_OF_INTEREST seconds vs. distance from magnet.
    0 time_margin means exact time only. Non-zero margin instructs to take max in that window,
    which is useful because there is jitter in the measurement
    (sometimes particles seem to disappear for a second).
    """
    substance_mass_rpm_combos = get_substance_mass_rpm_combos(path_to_videos, config)
    for smr in substance_mass_rpm_combos:
        fig, ax = plt.subplots()
        plot_height_at_time_of_interest_by_distance(
            path_to_videos, config, metadata, smr[0], smr[1], smr[2], ax
        )
        fig.tight_layout()
        plt.savefig(
            get_height_by_dist_at_time_fpath(
                config, smr[0], smr[1], smr[2], config.time_of_interest_sec
            )
        )
        plt.close(fig)


def plot_all_side_by_side_coated_vs_uncoated(
    path_to_videos: str | pathlib.Path, config: Config, metadata: dict[str, str]
):
    def generate_matches(substance, mass, rpm, distance=None):
        for vpath, vd, _ in generate_video_paths(path_to_videos, config):
            if (
                substance == substance
                and vd.mass == mass
                and vd.rpm == rpm
                and (vd.distance == distance or distance is None)
            ):
                yield vpath

    smrs = get_substance_mass_rpm_combos(path_to_videos, config)
    uncoated_smrs = [smr for smr in smrs if not smr[0].lower().startswith("si")]
    for substance, mass, rpm in uncoated_smrs:
        uncoated_vpaths = list(generate_matches(substance, mass, rpm))
        if len(uncoated_vpaths) > 1:
            logger.warning(
                f"Multiple uncoated videos found for {substance} {mass} {rpm}."
                " They will all compare against the same coated video."
            )
        for uncoated_vpath in uncoated_vpaths:
            vd = video_path_to_run_details(uncoated_vpath, metadata)
            coated_sub = f"Si{substance}"
            coated_vpaths = list(generate_matches(coated_sub, mass, rpm, vd.distance))
            if not coated_vpaths:
                continue
            if len(coated_vpaths) > 1:
                print(
                    f"Warning: multiple coated videos found for {coated_sub} {mass} {rpm}."
                    " Using first one."
                )
            coated_vpath = coated_vpaths[0]
            fig = plt.figure()
            gs = GridSpec(3, 6, figure=fig)
            axs = [
                # Left four: full height
                fig.add_subplot(gs[:, 0]),  # column 0, rows 0–2
                fig.add_subplot(gs[:, 1]),  # column 1, rows 0–2
                fig.add_subplot(gs[:, 2]),  # column 2, rows 0–2
                fig.add_subplot(gs[:, 3]),  # column 3, rows 0–2
                # Right two: stacked (1/3 height each)
                fig.add_subplot(gs[0, 4]),
                fig.add_subplot(gs[0, 5]),
                fig.add_subplot(gs[1, 4]),
                fig.add_subplot(gs[1, 5]),
                fig.add_subplot(gs[2, 4]),
                fig.add_subplot(gs[2, 5]),
            ]
            uncoated_quant_path = get_le_lage_fpath(uncoated_vpath, config, metadata)
            coated_quant_path = get_le_lage_fpath(coated_vpath, config, metadata)
            if not all(
                [
                    p.exists()
                    for p in [
                        uncoated_vpath,
                        coated_vpath,
                        uncoated_quant_path,
                        coated_quant_path,
                    ]
                ]
            ):
                continue
            plot_distribution_by_time(uncoated_vpath, config, metadata, axs[0])
            plot_distribution_by_time(coated_vpath, config, metadata, axs[1])
            plot_video_as_image(uncoated_vpath, config, metadata, axs[2])
            plot_video_as_image(coated_vpath, config, metadata, axs[3])
            plot_distribution(
                uncoated_vpath, config, metadata, config.time_of_interest_sec, axs[4]
            )
            plot_distribution(
                coated_vpath, config, metadata, config.time_of_interest_sec, axs[5]
            )
            plot_height_over_time(
                uncoated_quant_path,
                config,
                metadata,
                axs[6],
                title_includes_magnet_distance=False,
            )
            plot_height_over_time(
                coated_quant_path,
                config,
                metadata,
                axs[7],
                title_includes_magnet_distance=False,
            )
            plot_height_at_time_of_interest_by_distance(
                path_to_videos, config, metadata, substance, mass, rpm, axs[8]
            )
            plot_height_at_time_of_interest_by_distance(
                path_to_videos, config, metadata, coated_sub, mass, rpm, axs[9]
            )
            maximize_figure()
            fig.tight_layout()
            plt.savefig(
                get_coated_vs_uncoated_plot_path(
                    config, substance, mass, rpm, vd.distance
                )
            )
            plt.close(fig)


APP = typer.Typer()


@APP.command()
def main(
    videos_path: Annotated[
        pathlib.Path,
        typer.Argument(help="Path to directory with videos to process."),
    ],
    config_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to TOML config containing default options and configuration for project"
        ),
    ] = pathlib.Path(__file__).parent.parent.parent / "config.toml",
    metadata_path: Annotated[
        pathlib.Path, typer.Argument(help="Path to json metadata cache")
    ] = pathlib.Path(__file__).parent.parent.parent / "video_metadata.json",
    compare: Annotated[
        bool,
        typer.Option(
            help="Whether to plot coated and uncoated side-by-side. If specified, will overwrite existing quantification and plots."
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(help="Output debug information"),
    ] = False,
    log_file: Annotated[pathlib.Path, typer.Argument(help="File to stream logs into")]
    | None = None,
) -> None:
    """
    Quantify particle movement in videos.
    """
    # Set logging level for application
    if log_file:
        logging.basicConfig(filename=log_file, format="%(levelname)s - %(message)s\n")
    else:
        logging.basicConfig(format="%(levelname)s - %(message)s\n")

    logger.root.setLevel(logging.DEBUG if debug else logging.INFO)

    # Load config
    try:
        config: Config = Config.from_toml(config_path)
    except FileNotFoundError:
        logger.warning(
            f"TOML configuration file provided ({config_path}) not found. Trying from default configuration and moving on..."
        )
        config: Config = Config()
    except Exception:
        logger.error(
            "Unknown exception occured while reading TOML configuration file. Exiting..."
        )
        raise

    # Write metadata
    metadata = {}
    for _, _, ind_metadata in generate_video_paths(videos_path, config):
        metadata |= ind_metadata

    metadata_path.write_text(json.dumps(metadata, indent=4))

    normalize_videos(videos_path, config, metadata)

    quantify_normalized_videos(videos_path, config, metadata)
    plot_height_over_time_all_videos(videos_path, config, metadata)
    plot_all_height_over_time_all_videos(videos_path, config, metadata)
    plot_height_at_time_of_interest_by_distance_all_videos(
        videos_path, config, metadata
    )
    if compare:
        plot_all_side_by_side_coated_vs_uncoated(videos_path, config, metadata)


if __name__ == "__main__":
    typer.run(main)

```

## FILE: config.py

```python
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    @classmethod
    def from_toml(cls, toml: Path) -> "Config":
        with open(toml, "rb") as fid:
            config_dict = tomllib.load(fid)

        return cls(**config_dict)

    root_dir: Path = Path(
        "G:/Shared drives/Theralode/theralode-exp Drafts/EXP_TBD_NMR IONP"
    )
    save_dir: Path = Path(__file__).parent.parent.parent / "output"
    norm_videos_dir: Path = save_dir / "normalized_videos"

    runs_to_quantify: list[str] = field(
        default_factory=lambda: [
            "20251212 Runs",
            "20251210 Runs",  # ...
        ]
    )

    video_suffixes: list[str] = field(default_factory=lambda: [".wmv", ".mp4", ".avi"])

    time_of_interest_sec: int = 120
    num_secs_to_process: int = 300

    bw_threshold_default: float = 45.0
    px_per_mm: float = (541 - 246) / 42.0

    rewrite_if_exists: bool = False

    magnet_distance_mm: dict[int, int] = field(
        default_factory=lambda: {0: 74, 1: 89, 2: 104, 3: 119, 4: 134, 5: 149, 6: 164}
    )

    frames_per_real_sec: dict[str, float] = field(
        default_factory=lambda: {
            "2025-12-12": 3.7,
            "2025-12-10": 3.7,  # ...
        }
    )
    default_frames_per_real_sec: float = 3.7

    empty_tube_video_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "empty_tube.wmv"
    )

```

## FILE: file_names_and_paths.py

```python
"""
File and path utilities for particle analysis.
"""

import pathlib
from dataclasses import dataclass
from typing import Generator

import pandas as pd

from nmr_particle_motion.config import Config


@dataclass
class RunDetails:
    substance: str
    mass: str
    rpm: str
    distance: int
    datestamp: str

    @classmethod
    def from_dict(cls, d: dict) -> "RunDetails":
        datestamp = d.get("datestamp", "")
        return cls(
            substance=d.get("substance", ""),
            mass=d.get("mass", ""),
            rpm=d.get("rpm", ""),
            distance=int(d.get("distance", 0)),
            datestamp=datestamp,
        )


def _parse_filename(path: str | pathlib.Path) -> RunDetails:
    """Parse the filename to extract run details.
    Expected format: '<Substance> <Mass> <RPM> <Distance>___<Datestamp>'
    Example: 'IONP 10mg 3000rpm 3cm__20251212.suffix_is_ignored'
    """
    stem = pathlib.Path(path).stem
    try:
        if "__" in stem:
            if stem.startswith("Mag "):
                stem = (
                    stem[:3] + "_" + stem[4:]
                )  # Replace the first ' ' after 'Mag' with '_'
            name, datestamp = stem.split("__")  # separate datestamp
            datestamp = datestamp.strip().strip("_").strip()  # Clean datestamp
            parts = (
                name.strip().replace("_", " ").split(" ")
            )  # separate other parts, replacing _ with space just-in-case
            mass, rpm, distance = parts[-3], parts[-2], parts[-1]
            substance = name.split(mass)[0].strip()
            return RunDetails(substance, mass, rpm, int(distance[1:]), datestamp)
    except Exception:
        pass  # Fall through to default for empty tube
    return RunDetails("", "", "", 0, "empty_tube")


def get_normalized_video_fpath(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> pathlib.Path:
    """Returns path to corresponding normalized video."""
    stem = video_path.stem.removesuffix("_normalized")
    subdir_name = pathlib.Path(metadata[stem]).parent.name
    p = config.norm_videos_dir / subdir_name / f"{stem}_normalized.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_prenormalized_video_fpath(
    video_path: pathlib.Path, metadata: dict[str, str]
) -> pathlib.Path:
    """Returns path to corresponding "raw" (not normalized) video."""
    ends_to_trim = [
        "_normalized",
        "_le_lage",
        "_dist_over_time",
        "_height_over_time",
        "_le_lage_height_over_time",
    ]

    stem = video_path.stem
    for end in ends_to_trim:
        stem = stem.removesuffix(end)

    if stem not in metadata:
        return video_path  # Assume it's prenormalized.

    return pathlib.Path(metadata[stem])


def video_path_to_run_details(
    path: str | pathlib.Path, metadata: dict[str, str]
) -> RunDetails:
    path = get_prenormalized_video_fpath(
        pathlib.Path(path), metadata
    )  # Ensure we have the original video path
    csv_path = path.parent / "run_details.csv"
    if not csv_path.is_file():
        # No CSV file, attempt to parse filename
        return _parse_filename(path)
    all_run_details = pd.read_csv(csv_path, header=0, index_col=0)
    return RunDetails.from_dict(all_run_details.loc[path.name].to_dict())


def generate_video_paths(
    path_to_videos: str | pathlib.Path,
    config: Config,
) -> Generator[tuple[pathlib.Path, RunDetails, dict[str, str]]]:
    """Generate paths to all video files to be processed."""
    run_dirs = [pathlib.Path(path_to_videos)]

    metadata: dict[str, str] = {}
    for d in run_dirs:
        if not d.is_dir():
            continue

        for suffix in config.video_suffixes:
            for video_path in d.glob(f"*{suffix}", case_sensitive=False):
                if not video_path.is_file():
                    continue

                metadata[video_path.stem] = video_path.expanduser().resolve().as_posix()
                yield (
                    video_path,
                    video_path_to_run_details(video_path, metadata),
                    metadata,
                )


def get_le_lage_fpath(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> pathlib.Path:
    """Get the path to the quantifications CSV file.
    This is used to store the quantification results.
    """
    stem = video_path.stem.removesuffix("_normalized")
    subdir_name = pathlib.Path(metadata[stem]).parent.name
    p = config.save_dir / "leading_lagging_edges" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_le_lage.csv"


def get_height_over_time_fpath(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> pathlib.Path:
    """Get the path to the height over time CSV file.
    This is used to store the height over time results.
    """
    stem = video_path.stem.removesuffix("_normalized")
    subdir_name = pathlib.Path(metadata[stem]).parent.name
    p = config.save_dir / "height_over_time" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_le_lage_height_over_time.csv"


def get_full_video_distributions_fpaths(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str]
) -> tuple[pathlib.Path, pathlib.Path]:
    """Get the path to the full video distributions CSV file
    and corresponding distribution plot image.
    """
    stem = video_path.stem.removesuffix("_normalized")
    subdir_name = pathlib.Path(metadata[stem]).parent.name
    p = config.save_dir / "full_video_distributions" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_dist_over_time.csv", p / f"{stem}_distribution.png"


def get_dist_at_time_plot_fpath(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], time_sec: int
) -> pathlib.Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    stem = video_path.stem.removesuffix("_normalized")
    subdir_name = pathlib.Path(metadata[stem]).parent.name
    p = config.save_dir / f"{int(time_sec)}s_distributions" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{video_path.stem}_{int(time_sec)}s.png"


def get_height_by_dist_at_time_fpath(
    config: Config, substance, mass, rpm, time_sec: int
) -> pathlib.Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    p = (
        config.save_dir
        / f"{config.time_of_interest_sec}s_height_by_distance_from_magnet"
    )
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{substance}_{mass}_{rpm}_leading_edge_at_{time_sec}_seconds.png"


def get_coated_vs_uncoated_plot_path(config: Config, substance, mass, rpm, distance):
    p = config.save_dir / "coated_vs_uncoated_side_by_side"
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{substance}_{mass}_{rpm}_d{distance}_coated_vs_uncoated_side_by_side.png"
    )

```

## FILE: frame_generator.py

```python
"""
Frame generation utilities for particle analysis.
"""

import pathlib
import logging
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


logger = logging.getLogger("nmr_particle_motion")


@dataclass
class VideoData:
    """Data class to hold video and its metadata."""

    video: cv2.VideoCapture
    video_path: pathlib.Path
    nframes: int
    fps: float

    @classmethod
    def from_path(cls, video_path: pathlib.Path) -> "VideoData":
        """Create VideoData from a video file path.
        The video is opened and metadata is extracted.
        The video object is not released here; it should be released by the caller.
        """
        video = cv2.VideoCapture(video_path.as_posix())
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        return cls(video=video, video_path=video_path, nframes=nframes, fps=fps)


def generate_grayscale_frames(
    path_to_video: pathlib.Path,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Generate frames from the video file.
    Yields the frame index and the frame itself as a numpy array.
    """
    vd = VideoData.from_path(path_to_video)
    for i in range(vd.nframes):
        not_at_end, frame = vd.video.read()

        if frame is None:
            logger.debug(
                f"Unable to convert frame {i} of {vd.nframes} of video {vd.video_path} to grayscale."
            )

        if frame is not None and not_at_end:
            yield i, np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            logger.debug(
                f"Converted {i - 1} frames of {vd.nframes} for video {vd.video_path} to grayscale"
            )
            break

    # Release the video object
    vd.video.release()

```

## FILE: helpers_to_check_calculations.py

```python
"""Helpers to check calculations used in particle analysis."""

import pathlib

from matplotlib import pyplot as plt

from nmr_particle_motion.frame_generator import (
    generate_grayscale_frames,
)
from nmr_particle_motion.globals import (
    NUM_SECS_TO_PROCESS_IN_VIDEO,
    ROOT_DIR,
    RUNS_TO_QUANTIFY,
)
from nmr_particle_motion.unit_conversions import (
    frame_to_time,
    time_to_frames,
)
from nmr_particle_motion.video_normalizing import (
    VideoNormalizer,
    plot_frame,
)


def inspect_frame_time_conversion():
    """
    Check whether the frame-to-time and time-to-frame conversions are accurate.
    """
    for run_subdir in RUNS_TO_QUANTIFY[2:]:
        run_dir = ROOT_DIR / run_subdir
        if not run_dir.is_dir():
            continue
        video_path = list(run_dir.glob("*.wmv", case_sensitive=False))[0]
        frames_of_interest = [
            time_to_frames(video_path, t) for t in [0, NUM_SECS_TO_PROCESS_IN_VIDEO]
        ]
        for frame_i, frame in generate_grayscale_frames(video_path):
            if frame_i in frames_of_interest:
                time = frame_to_time(video_path, frame_i)
                plot_frame(frame, f"Frame {frame_i} at time {time:.2f} seconds")
        plt.show()


def inspect_binarization_parameters(vpath: pathlib.Path):
    """Visualize the effect of different binarization parameters on a sample frame."""
    vn = VideoNormalizer(vpath)
    for frame_i, frame in generate_grayscale_frames(vpath):
        if frame_i == time_to_frames(vpath, 0):
            vn.tune_binarization_parameters(frame, show=True)
            break


if __name__ == "__main__":
    inspect_binarization_parameters(
        ROOT_DIR
        / "20251209 Runs"
        / "3E1 2000ug 100rpm d2___2025-12-09_15-01-29-4307-06-00.wmv"
    )
    inspect_binarization_parameters(
        ROOT_DIR / "20251215 Runs" / "Settle___2025-12-15_13-14-14-1649-06-00.wmv"
    )

```

## FILE: leading_lagging_edge_and_particle_dist.py

```python
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

```

## FILE: plot_output.py

```python
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
from nmr_particle_motion.frame_generator import (
    generate_grayscale_frames,
)
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

```

## FILE: shapeglobals.py

```python
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

```

## FILE: unit_conversions.py

```python
"""Unit conversion functions for particle analysis."""

import pathlib

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    video_path_to_run_details,
)


def frame_to_time(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], frame_index: int
) -> float:
    """Get the time in seconds for the given frame index."""
    datestamp = (
        video_path_to_run_details(video_path, metadata)
        .datestamp.strip("_")
        .split("_")[0]
    )
    frames_per_rt_secs = config.frames_per_real_sec.get(
        datestamp, config.default_frames_per_real_sec
    )
    return frame_index / frames_per_rt_secs


def time_to_frames(
    video_path: pathlib.Path, config: Config, metadata: dict[str, str], time_sec: int
) -> int:
    """Convert time in seconds to number of frames."""
    datestamp = (
        video_path_to_run_details(video_path, metadata)
        .datestamp.strip("_")
        .split("_")[0]
    )
    frames_per_rt_secs = config.frames_per_real_sec.get(
        datestamp, config.default_frames_per_real_sec
    )
    return int(time_sec * frames_per_rt_secs)


def get_mm_distance(config: Config, px_distance: float | int) -> float:
    """Convert pixel distance to mm distance."""
    return px_distance / config.px_per_mm

```

## FILE: video_normalizing.py

```python
import logging
import pathlib

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    get_normalized_video_fpath,
    get_prenormalized_video_fpath,
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

    def __init__(self, vpath: pathlib.Path, config: Config, metadata: dict[str, str]):
        # Just to assert it's a pre-normalized video.
        self.video_path = get_prenormalized_video_fpath(vpath, metadata)

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
        for i, frame in generate_grayscale_frames(null_vpath):
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

    def normalize_video(self, config: Config, metadata: dict[str, str]) -> None:
        """Perform the quantification of bead mixing in the video."""
        norm_video_path = get_normalized_video_fpath(self.video_path, config, metadata)
        if norm_video_path.exists() and not config.rewrite_if_exists:
            logger.info(
                f"Normalized video {norm_video_path} already exists. Skipping normalization."
            )
            return
        is_first = True
        with VideoWriter(
            outpath=norm_video_path,
            fps=30,
            width=self.SG.FINAL_FRAME_SHAPE[1],
            height=self.SG.FINAL_FRAME_SHAPE[0],
            overwrite=True,
            save_all_data=True,
        ) as video_writer:
            for _, frame in tqdm(generate_grayscale_frames(self.video_path)):
                if is_first:
                    self.tune_brightness_scale_factor(frame)
                    # self.tune_binarization_parameters(frame)
                    is_first = False
                normalized_frame = self.normalize_frame(frame)
                quant_frame = normalized_frame.copy()
                quant_frame[~self.SG.QUANT_MASK] = 0
                video_writer.write_frame(quant_frame)

```

## FILE: video_writer.py

```python
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

```

