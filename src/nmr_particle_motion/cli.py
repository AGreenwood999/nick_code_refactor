"""Quantify bead mixing in a video."""

import logging
import pathlib
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from tqdm import tqdm

from nmr_particle_motion.config import Config
from nmr_particle_motion.file_names_and_paths import (
    VideoContext,
    get_all_video_contexts_from_directory,
    get_dist_at_time_plot_fpath,
    get_full_video_distributions_fpaths,
    get_height_over_time_fpath,
    get_le_lage_fpath,
)
from nmr_particle_motion.leading_lagging_edge_and_particle_dist import (
    quantify_normalized_video,
)
from nmr_particle_motion.plot_output import (
    plot_distribution,
    plot_distribution_by_time,
    plot_height_over_time,
)
from nmr_particle_motion.video_normalizing import VideoNormalizer

logger = logging.getLogger("nmr_particle_motion")


def maximize_figure() -> None:
    """Set the size of a fullscreen figure."""
    mgr = plt.get_current_fig_manager()
    if mgr is not None:
        mgr.full_screen_toggle()


def save_distribution_plot(
    ctx: VideoContext,
    config: Config,
    time_sec: int,
    show: bool,
) -> None:
    """Plot the distribution of particles at a given time.
    This method visualizes the quantification results.
    """
    plot_fpath = get_dist_at_time_plot_fpath(ctx, config, time_sec)
    if plot_fpath.exists() and not config.overwrite_plots:
        logger.info(
            f"Distribution file {plot_fpath} already exists. Skipping distribution plotting."
        )
        return
    fig, ax = plt.subplots()
    plot_distribution(ctx, config, time_sec, ax)
    ax.grid(visible=True, which="both", axis="both", color="r", linewidth=2)
    fig.tight_layout()
    plt.savefig(plot_fpath)
    if show:
        plt.show()
    plt.close(fig)


def save_distribution_by_time_plot(
    ctx: VideoContext, config: Config, show: bool
) -> None:
    """Plot the distribution of particles at a given time.
    This method visualizes the quantification results.
    """
    fpath, plot_fpath = get_full_video_distributions_fpaths(ctx, config)
    if fpath.exists() and plot_fpath.exists() and not config.overwrite_plots:
        logger.info(
            f"Distribution file {fpath} already exists. Skipping distribution plotting."
        )
        return
    fig, ax = plt.subplots()
    distributions = plot_distribution_by_time(ctx, config, ax)
    ax.grid(visible=True, which="both", axis="both", color="r", linewidth=2)
    fig.tight_layout()
    plt.savefig(plot_fpath)
    if show:
        plt.show()
    plt.close(fig)
    with open(fpath, "wt", encoding="utf-8") as ofh:
        pd.DataFrame(distributions).to_csv(
            ofh, header=False, index=False, lineterminator="\n"
        )


def normalize_video(ctx: VideoContext, config: Config):
    VideoNormalizer(ctx, config).normalize_video(config)


def quantify_normalized_videos(contexts: list[VideoContext], config: Config) -> None:
    for ctx in tqdm(contexts):
        logging.info(f"Quantifying run {ctx.path.stem}")
        quantify_normalized_video(ctx, config)
        save_distribution_plot(
            ctx, config, time_sec=config.time_of_interest_sec, show=False
        )
        save_distribution_by_time_plot(ctx, config, show=False)


def plot_height_over_time_all_videos(contexts: list[VideoContext], config: Config):
    for ctx in contexts:
        quant_path = get_le_lage_fpath(ctx, config)
        fig, ax = plt.subplots()
        plot_height_over_time(quant_path, ctx, config, ax)
        ax.grid(visible=True, which="both", axis="both", color="r", linewidth=2)
        ax.legend()
        fig.tight_layout()
        plt.savefig(get_height_over_time_fpath(ctx, config).with_suffix(".png"))
        plt.close(fig)


def plot_all_height_over_time_all_videos(contexts: list[VideoContext], config: Config):
    fig, axs = plt.subplots(
        int(np.ceil(len(contexts) / 3)), 3, sharex=True, sharey=True
    )
    axs = axs.flatten()
    for i, ctx in enumerate(contexts):
        axs[i].grid(visible=True, which="both", axis="both", color="r", linewidth=2)
        quant_path = get_le_lage_fpath(ctx, config)
        plot_height_over_time(quant_path, ctx, config, axs[i])
    fig.suptitle("Particle Height Over Time")
    maximize_figure()
    fig.tight_layout()
    grandparent = get_height_over_time_fpath(ctx, config).parent.parent  # type:ignore
    figpath = grandparent / "all_height_over_time.png"
    if figpath.exists():
        logging.warning(f"{figpath} will be overwritten.")
    plt.savefig(figpath)
    plt.close(fig)


def plot_all_side_by_side_coated_vs_uncoated(*args):
    pass


"""
def plot_all_side_by_side_coated_vs_uncoated(
    contexts: list[VideoContext], config: Config
):
    uncoated_smrs = [smr for smr in smrs if not smr[0].lower().startswith("si")]
    for substance, mass, rpm in uncoated_smrs:
        uncoated_vpaths = list(generate_matches(substance, mass, rpm))
        if len(uncoated_vpaths) > 1:
            logger.warning(
                f"Multiple uncoated videos found for {substance} {mass} {rpm}."
                " They will all compare against the same coated video."
            )
        for uncoated_vpath in uncoated_vpaths:
            vd = get_run_details_from_path_csv(uncoated_vpath, metadata)
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
"""

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
        logging.basicConfig(filename=log_file, format="\n%(levelname)s - %(message)s\n")
    else:
        logging.basicConfig(format="\n%(levelname)s - %(message)s")

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

    contexts = get_all_video_contexts_from_directory(videos_path, config)

    for ctx in contexts:
        normalize_video(ctx, config)

    quantify_normalized_videos(contexts, config)
    plot_height_over_time_all_videos(contexts, config)
    plot_all_height_over_time_all_videos(contexts, config)

    if compare:
        plot_all_side_by_side_coated_vs_uncoated(contexts, config)


if __name__ == "__main__":
    typer.run(main)
