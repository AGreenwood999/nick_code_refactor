"""
File and path utilities for particle analysis.
"""

import json
import pathlib
from dataclasses import dataclass

import pandas as pd

from nmr_particle_motion.particle_analysis_lib.globals import (
    NORM_VIDEOS_DIR,
    ROOT_DIR,
    RUNS_TO_QUANTIFY,
    SAVE_DIR,
    TIME_OF_INTEREST,
    VIDEO_METADATA_CACHE_PATH,
    VIDEO_PATH_SUFFIXES,
)

__VIDEO_METADATA_CACHE: dict[str, str] = {}


@dataclass
class RunDetails:
    substance: str
    mass: str
    rpm: str
    distance: int
    datestamp: str
    has_mm_scale: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "RunDetails":
        datestamp = d.get("datestamp", "")
        has_mm_scale = bool(
            d.get("has_mm_scale", datestamp.replace("-", "") <= "20251213")
        )  # MM scale removed after 2025-12-12
        return cls(
            substance=d.get("substance", ""),
            mass=d.get("mass", ""),
            rpm=d.get("rpm", ""),
            distance=int(d.get("distance", 0)),
            datestamp=datestamp,
            has_mm_scale=has_mm_scale,
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
            has_mm_scale = (
                datestamp.replace("-", "") <= "20251213"
            )  # MM scale removed after 2025-12-12
            return RunDetails(
                substance, mass, rpm, int(distance[1:]), datestamp, has_mm_scale
            )
    except Exception:
        pass  # Fall through to default for empty tube
    has_mm_scale = stem == "Empty Tube 20251204"
    return RunDetails("", "", "", 0, "empty_tube", has_mm_scale)


def get_normalized_video_fpath(video_path: pathlib.Path) -> pathlib.Path:
    """Returns path to corresponding normalized video."""
    global __VIDEO_METADATA_CACHE
    if not __VIDEO_METADATA_CACHE and VIDEO_METADATA_CACHE_PATH.is_file():
        __VIDEO_METADATA_CACHE = json.loads(VIDEO_METADATA_CACHE_PATH.read_text())
    stem = video_path.stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    subdir_name = pathlib.Path(__VIDEO_METADATA_CACHE[stem]).parent.name
    p = NORM_VIDEOS_DIR / subdir_name / f"{stem}_normalized.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_prenormalized_video_fpath(video_path: pathlib.Path) -> pathlib.Path:
    """Returns path to corresponding "raw" (not normalized) video."""
    global __VIDEO_METADATA_CACHE
    if not __VIDEO_METADATA_CACHE and VIDEO_METADATA_CACHE_PATH.is_file():
        __VIDEO_METADATA_CACHE = json.loads(VIDEO_METADATA_CACHE_PATH.read_text())
    stem = video_path.stem
    ends_to_trim = [
        "_normalized",
        "_le_lage",
        "_dist_over_time",
        "_height_over_time",
        "_le_lage_height_over_time",
    ]
    for end in ends_to_trim:
        if stem.endswith(end):
            stem = stem[: -len(end)]
    if stem not in __VIDEO_METADATA_CACHE:
        return video_path  # Assume it's prenormalized.
    return pathlib.Path(__VIDEO_METADATA_CACHE[stem])


def video_path_to_run_details(path: str | pathlib.Path) -> RunDetails:
    """Assume that __VIDEO_METADATA_CACHE exists and is filled."""
    path = get_prenormalized_video_fpath(
        pathlib.Path(path)
    )  # Ensure we have the original video path
    csv_path = path.parent / "run_details.csv"
    if not csv_path.is_file():
        # No CSV file, attempt to parse filename
        return _parse_filename(path)
    all_run_details = pd.read_csv(csv_path, header=0, index_col=0)
    return RunDetails.from_dict(all_run_details.loc[path.name].to_dict())


def generate_video_paths(path_to_videos: str | pathlib.Path | None = None):
    """Generate paths to all video files to be processed."""
    global __VIDEO_METADATA_CACHE
    if not __VIDEO_METADATA_CACHE and VIDEO_METADATA_CACHE_PATH.is_file():
        __VIDEO_METADATA_CACHE = json.loads(VIDEO_METADATA_CACHE_PATH.read_text())
    if path_to_videos is None:
        run_dirs = [ROOT_DIR / run_subdir for run_subdir in RUNS_TO_QUANTIFY]
    else:
        run_dirs = [pathlib.Path(path_to_videos)]
    for d in run_dirs:
        if not d.is_dir():
            continue
        for suffix in VIDEO_PATH_SUFFIXES:
            for vp in d.glob(f"*{suffix}", case_sensitive=False):
                if not vp.is_file():
                    continue
                __VIDEO_METADATA_CACHE[vp.stem] = vp.expanduser().resolve().as_posix()
                VIDEO_METADATA_CACHE_PATH.write_text(
                    json.dumps(__VIDEO_METADATA_CACHE, indent=2)
                )  # Update cache file
                yield vp, video_path_to_run_details(vp)


def get_le_lage_fpath(video_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the quantifications CSV file.
    This is used to store the quantification results.
    """
    stem = video_path.stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    subdir_name = pathlib.Path(__VIDEO_METADATA_CACHE[stem]).parent.name
    p = SAVE_DIR / "leading_lagging_edges" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_le_lage.csv"


def get_height_over_time_fpath(video_path: pathlib.Path) -> pathlib.Path:
    """Get the path to the height over time CSV file.
    This is used to store the height over time results.
    """
    stem = video_path.stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    subdir_name = pathlib.Path(__VIDEO_METADATA_CACHE[stem]).parent.name
    p = SAVE_DIR / "height_over_time" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_le_lage_height_over_time.csv"


def get_full_video_distributions_fpaths(
    video_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Get the path to the full video distributions CSV file
    and corresponding distribution plot image.
    """
    stem = video_path.stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    subdir_name = pathlib.Path(__VIDEO_METADATA_CACHE[stem]).parent.name
    p = SAVE_DIR / "full_video_distributions" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_dist_over_time.csv", p / f"{stem}_distribution.png"


def get_dist_at_time_plot_fpath(
    video_path: pathlib.Path, time_sec: int
) -> pathlib.Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    stem = video_path.stem
    if stem.endswith("_normalized"):
        stem = stem[: -len("_normalized")]
    subdir_name = pathlib.Path(__VIDEO_METADATA_CACHE[stem]).parent.name
    p = SAVE_DIR / f"{int(time_sec)}s_distributions" / subdir_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{video_path.stem}_{int(time_sec)}s.png"


def get_height_by_dist_at_time_fpath(
    substance, mass, rpm, time_sec: int
) -> pathlib.Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    p = SAVE_DIR / f"{TIME_OF_INTEREST}s_height_by_distance_from_magnet"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{substance}_{mass}_{rpm}_leading_edge_at_{time_sec}_seconds.png"


def get_coated_vs_uncoated_plot_path(substance, mass, rpm, distance):
    p = SAVE_DIR / "coated_vs_uncoated_side_by_side"
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{substance}_{mass}_{rpm}_d{distance}_coated_vs_uncoated_side_by_side.png"
    )


def video_has_mm_scale(path: pathlib.Path) -> bool:
    """Determine if the video has a mm scale based on its filename."""
    rd = video_path_to_run_details(path)
    return rd.has_mm_scale
