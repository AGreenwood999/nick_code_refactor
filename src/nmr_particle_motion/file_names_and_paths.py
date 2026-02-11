"""
File and path utilities for particle analysis.
"""

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from nmr_particle_motion.config import Config

import logging

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


def get_video_details_from_path(path: Path) -> RunDetails:
    csv_path = path.parent / "run_details.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"Run details CSV not found at {csv_path}")

    all_run_details = pd.read_csv(csv_path, header=0)
    all_run_details.set_index("name", inplace=True)
    return RunDetails.from_dict(all_run_details.loc[path.name].to_dict(), path.stem)


def get_all_video_contexts_from_directory(
    videos_dir: Path, config: Config
) -> list[VideoContext]:
    contexts = []
    for suffix in config.video_suffixes:
        for video_path in videos_dir.glob(f"*{suffix}", case_sensitive=False):
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
                )
            )

    return contexts


def get_le_lage_fpath(ctx: VideoContext, config: Config) -> Path:
    """Get the path to the quantifications CSV file.
    This is used to store the quantification results.
    """
    p = config.save_dir / Path("leading_lagging_edges")
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ctx.details.lot}-{ctx.details.vial_number}_le_lage.csv"


def get_height_over_time_fpath(ctx: VideoContext, config: Config) -> Path:
    """Get the path to the height over time CSV file.
    This is used to store the height over time results.
    """
    p = config.save_dir / Path("height_over_time")
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{ctx.details.lot}-{ctx.details.vial_number}_le_lage_height_over_time.csv"
    )


def get_full_video_distributions_fpaths(
    ctx: VideoContext, config: Config
) -> tuple[Path, Path]:
    """Get the path to the full video distributions CSV file
    and corresponding distribution plot image.
    """
    p = config.save_dir / Path("full_video_distributions")
    p.mkdir(parents=True, exist_ok=True)
    return (
        p / f"{ctx.details.lot}-{ctx.details.vial_number}_dist_over_time.csv",
        p / f"{ctx.details.lot}-{ctx.details.vial_number}_distribution.png",
    )


def get_dist_at_time_plot_fpath(
    ctx: VideoContext, config: Config, time_sec: int
) -> Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    p = config.save_dir / Path(f"{int(time_sec)}s_distributions")
    p.mkdir(parents=True, exist_ok=True)
    if ctx.details.lot == "Unknown lot" or ctx.details.vial_number == -1:
        logger.warning(
            f"Unable to parse lot or vial number from file {ctx.details.name}. Defaulting to filename"
        )
        return p / f"{ctx.details.lot}-{ctx.details.vial_number}_{int(time_sec)}s.png"
    else:
        return p / f"{ctx.details.lot}-{ctx.details.vial_number}_{int(time_sec)}s.png"


def get_height_by_dist_at_time_fpath(
    config: Config, substance, mass, rpm, time_sec: int
) -> Path:
    """Get the path to the time-of-interest CSV file
    and corresponding distribution plot image.
    """
    p = config.save_dir / Path(
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
