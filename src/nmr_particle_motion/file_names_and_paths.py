"""
File and path utilities for particle analysis.
"""

import pathlib
from dataclasses import dataclass
from typing import Generator

from cv2.gapi import video
import pandas as pd

from nmr_particle_motion.config import Config


@dataclass(frozen=True)
class VideoContext:
    path: pathlib.Path
    norm_path: pathlib.Path
    details: RunDetails


@dataclass
class RunDetails:
    substance: str
    mass: str
    rpm: str
    distance: int
    datestamp: str

    @classmethod
    def from_dict(cls, d: dict) -> "RunDetails":
        return cls(
            substance=d.get("substance", ""),
            mass=d.get("mass", ""),
            rpm=d.get("rpm", ""),
            distance=int(d.get("distance", 0)),
            datestamp=d.get("datestamp", ""),
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
    all_run_details = pd.read_csv(csv_path, header=0, index_col="name")
    return RunDetails.from_dict(all_run_details.loc[path.name].to_dict())


def get_video_details_from_path(path: pathlib.Path) -> RunDetails:
    csv_path = path.parent / "run_details.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"Run details CSV not found at {csv_path}")

    all_run_details = pd.read_csv(csv_path, header=0, index_col=0)
    return RunDetails.from_dict(all_run_details.loc[path.name].to_dict())


def get_all_video_contexts_from_directory(
    videos_dir: pathlib.Path, config: Config
) -> Generator[VideoContext]:
    for suffix in config.video_suffixes:
        for video_path in videos_dir.glob(f"*{suffix}", case_sensitive=False):
            if not video_path.is_file():
                continue

            norm_video_path = video_path.with_name(
                f"{video_path.stem}_normalized{video_path.suffix}"
            )

            yield VideoContext(
                video_path.expanduser().resolve(),
                norm_video_path.expanduser().resolve(),
                get_video_details_from_path(video_path),
            )


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
    p = (
        config.save_dir
        / pathlib.Path("leading_lagging_edges")
        / pathlib.Path(subdir_name)
    )
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
    p = config.save_dir / pathlib.Path("height_over_time") / pathlib.Path(subdir_name)
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
    p = (
        config.save_dir
        / pathlib.Path("full_video_distributions")
        / pathlib.Path(subdir_name)
    )
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
    p = (
        config.save_dir
        / pathlib.Path(f"{int(time_sec)}s_distributions")
        / pathlib.Path(subdir_name)
    )
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{video_path.stem}_{int(time_sec)}s.png"


def get_height_by_dist_at_time_fpath(
    config: Config, substance, mass, rpm, time_sec: int
) -> pathlib.Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    p = config.save_dir / pathlib.Path(
        f"{config.time_of_interest_sec}s_height_by_distance_from_magnet"
    )
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{substance}_{mass}_{rpm}_leading_edge_at_{time_sec}_seconds.png"


def get_coated_vs_uncoated_plot_path(config: Config, substance, mass, rpm, distance):
    p = config.save_dir / "coated_vs_uncoated_side_by_side"
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{substance}_{mass}_{rpm}_d{distance}_coated_vs_uncoated_side_by_side.png"
    )
