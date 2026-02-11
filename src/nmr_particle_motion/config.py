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

    video_suffixes: tuple[str, str, str] = (".wmv", ".mp4", ".avi")

    time_of_interest_sec: int = 120
    num_secs_to_process: int = 300

    bw_threshold_default: float = 45.0
    px_per_mm: float = (541 - 246) / 42.0

    rewrite_vids: bool = False
    overwrite_norm_vid: bool = False
    overwrite_plots: bool = False
    overwrite_lead_lag_edge_data: bool = False

    magnet_distance_mm: dict[int, int] = field(
        default_factory=lambda: {0: 74, 1: 89, 2: 104, 3: 119, 4: 134, 5: 149, 6: 164}
    )

    frames_per_real_sec: float = 3.7

    empty_tube_video_path: Path = (
        Path(__file__).parent.parent.parent / "data" / "empty_tube.wmv"
    )
