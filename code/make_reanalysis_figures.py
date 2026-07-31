#!/usr/bin/env python3
"""Reproduce manuscript Figures 2 and 3 from independence-controlled results.

The script reads numerical outputs created by ``code/run_reanalysis.py`` and
produces PNG, PDF and SVG versions of the two data-derived manuscript figures.
No generative image model is used.  A fixed random seed affects only display
jitter and the deterministic subsample of PTB-XL points shown for readability;
all reported means and confidence intervals use the complete analysis outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLOURS = {
    "WESAD": "#3D8B67",
    "CASE": "#D9705A",
    "PTB-XL": "#4C78A8",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figures 2 and 3 from reanalysis outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results_independence_controlled",
        help=(
            "Directory created by run_reanalysis.py. Default: the "
            "results_independence_controlled directory next to github_release."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures",
        help="Directory for generated figure files. Default: repository figures/.",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--ptb-display-max",
        type=int,
        default=900,
        help=(
            "Maximum number of PTB-XL patient points displayed in Figure 2. "
            "Summary estimates always use all patients."
        ),
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required analysis outputs were not found:\n  - " + formatted +
            "\nRun code/run_reanalysis.py first or provide --results-dir."
        )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> None:
    for ext in formats:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(f".{ext}"), **kwargs)


def make_figure_2(
    root: Path,
    out: Path,
    summary: pd.DataFrame,
    rng: np.random.Generator,
    formats: Iterable[str],
    dpi: int,
    ptb_display_max: int,
) -> None:
    items = [
        (
            "WESAD",
            root / "wesad/primary_cross_subject/independent_unit_deltas.csv",
            "Participant-level Δ",
            (-0.05, 0.85),
        ),
        (
            "CASE",
            root / "case/primary_valence_arousal_quadrant/independent_unit_deltas.csv",
            "Participant-level Δ",
            (-0.06, 0.08),
        ),
        (
            "PTB-XL",
            root / "ptbxl/primary_patient_independent/independent_unit_deltas.csv",
            "Patient-level Δ",
            (-0.52, 0.60),
        ),
    ]
    require_files([item[1] for item in items])

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), constrained_layout=True)
    for index, (name, path, ylabel, ylim) in enumerate(items):
        ax = axes[index]
        frame = pd.read_csv(path)
        values = frame["delta"].to_numpy(float)

        parts = ax.violinplot(
            values,
            positions=[1],
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(DEFAULT_COLOURS[name])
            body.set_edgecolor(DEFAULT_COLOURS[name])
            body.set_alpha(0.22)
            body.set_linewidth(0.8)

        # PTB-XL contains 7,491 patients. A deterministic display subsample keeps
        # the plot readable. The mean and confidence interval below are computed
        # from the full patient-level analysis, not from the displayed subset.
        if name == "PTB-XL" and len(values) > ptb_display_max:
            displayed = rng.choice(values, size=ptb_display_max, replace=False)
        else:
            displayed = values
        jitter = rng.normal(0, 0.055, size=len(displayed))
        ax.scatter(
            np.full(len(displayed), 1.0) + jitter,
            displayed,
            s=12 if len(displayed) < 100 else 5,
            alpha=0.72 if len(displayed) < 100 else 0.30,
            color=DEFAULT_COLOURS[name],
            edgecolors="none",
            rasterized=len(displayed) > 100,
        )

        row = summary.loc[summary["dataset"] == name].iloc[0]
        mean = float(row["mean_unit_delta"])
        low = float(row["bootstrap_ci_95_low"])
        high = float(row["bootstrap_ci_95_high"])
        ax.errorbar(
            1.28,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            ms=5.6,
            capsize=3.5,
            color="black",
            ecolor="black",
            lw=1.2,
            zorder=5,
        )
        ax.axhline(0, color="#666666", lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0.48, 1.55)
        ax.set_ylim(*ylim)
        ax.set_xticks([1])
        ax.set_xticklabels([name])
        ax.set_ylabel(ylabel)
        n_units = int(row["n_independent_units"])
        ax.set_title(
            f"{chr(97 + index)}  {name}  (independent n = {n_units:,})",
            loc="left",
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.98,
            (
                f"mean Δ = {mean:.4f}\n"
                f"95% CI [{low:.4f}, {high:.4f}]\n"
                f"permutation P = {float(row['permutation_p_two_sided']):.4f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
                "linewidth": 0.7,
                "alpha": 0.94,
            },
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Independent-unit physiological similarity contrasts",
        fontsize=13.2,
        fontweight="bold",
    )
    save_all(fig, out / "Figure_independent_unit_deltas", formats, dpi)
    plt.close(fig)


def make_figure_3(
    root: Path,
    out: Path,
    summary: pd.DataFrame,
    rng: np.random.Generator,
    formats: Iterable[str],
    dpi: int,
) -> None:
    wesad_path = root / "wesad/sensitivity_latent_dimension_and_metric.csv"
    ptb_path = root / "ptbxl/sensitivity_latent_dimension_and_metric.csv"
    ptb_adjusted_path = root / "ptbxl/sensitivity_age_sex_adjusted/summary.csv"
    case_video_path = root / "case/sensitivity_video_centered/summary.csv"
    case_loo_path = root / "case/leave_one_video_out.csv"
    require_files(
        [wesad_path, ptb_path, ptb_adjusted_path, case_video_path, case_loo_path]
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.75), constrained_layout=True)

    ax = axes[0]
    wesad = pd.read_csv(wesad_path)
    for metric, marker, linestyle in (("cosine", "o", "-"), ("correlation", "s", "--")):
        data = wesad[wesad.metric == metric].sort_values("latent_dim")
        ax.plot(
            data.latent_dim,
            data.mean_unit_delta,
            marker=marker,
            ls=linestyle,
            lw=1.5,
            ms=5,
            label=metric.capitalize(),
        )
        ax.fill_between(
            data.latent_dim,
            data.bootstrap_ci_95_low,
            data.bootstrap_ci_95_high,
            alpha=0.12,
        )
    ax.axhline(0, color="#666666", ls="--", lw=0.8)
    ax.set_xticks([4, 8, 16])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean participant-level Δ")
    ax.set_title("a  WESAD embedding robustness", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ptb = pd.read_csv(ptb_path)
    for metric, marker, linestyle in (("cosine", "o", "-"), ("correlation", "s", "--")):
        data = ptb[ptb.metric == metric].sort_values("latent_dim")
        ax.plot(
            data.latent_dim,
            data.mean_unit_delta,
            marker=marker,
            ls=linestyle,
            lw=1.5,
            ms=5,
            label=metric.capitalize(),
        )
        ax.fill_between(
            data.latent_dim,
            data.bootstrap_ci_95_low,
            data.bootstrap_ci_95_high,
            alpha=0.12,
        )
    adjusted = pd.read_csv(ptb_adjusted_path).iloc[0]
    adjusted_mean = float(adjusted.mean_unit_delta)
    ax.errorbar(
        [8.7],
        [adjusted_mean],
        yerr=[
            [adjusted_mean - float(adjusted.bootstrap_ci_95_low)],
            [float(adjusted.bootstrap_ci_95_high) - adjusted_mean],
        ],
        fmt="D",
        color="#9C755F",
        capsize=3,
        ms=5,
        label="Age/sex adjusted",
    )
    ax.axhline(0, color="#666666", ls="--", lw=0.8)
    ax.set_xticks([4, 8, 16])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean patient-level Δ")
    ax.set_title("b  PTB-XL robustness", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    case_primary = summary.loc[summary.dataset == "CASE"].iloc[0]
    video = pd.read_csv(case_video_path).iloc[0]
    leave_one_out = pd.read_csv(case_loo_path)
    labels = ["Primary\ncross-video", "Video-centred", "Leave-one-video-out\nrange"]
    x = np.arange(3)
    primary_mean = float(case_primary.mean_unit_delta)
    ax.errorbar(
        x[0],
        primary_mean,
        yerr=[
            [primary_mean - float(case_primary.bootstrap_ci_95_low)],
            [float(case_primary.bootstrap_ci_95_high) - primary_mean],
        ],
        fmt="o",
        color=DEFAULT_COLOURS["CASE"],
        ms=6,
        capsize=4,
        lw=1.4,
    )
    video_mean = float(video.mean_unit_delta)
    ax.errorbar(
        x[1],
        video_mean,
        yerr=[
            [video_mean - float(video.bootstrap_ci_95_low)],
            [float(video.bootstrap_ci_95_high) - video_mean],
        ],
        fmt="s",
        color="#777777",
        ms=6,
        capsize=4,
        lw=1.4,
    )
    ax.vlines(
        x[2],
        leave_one_out.mean_unit_delta.min(),
        leave_one_out.mean_unit_delta.max(),
        color="#B279A2",
        lw=5,
        alpha=0.65,
    )
    ax.scatter(
        np.full(len(leave_one_out), x[2]) + rng.normal(0, 0.03, len(leave_one_out)),
        leave_one_out.mean_unit_delta,
        s=17,
        color="#B279A2",
        zorder=3,
    )
    ax.axhline(0, color="#666666", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean participant-level Δ")
    ax.set_title("c  CASE stimulus sensitivity", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        f"Video-centred estimate\nP = {float(video.permutation_p_two_sided):.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#BBBBBB",
            "linewidth": 0.7,
        },
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Robustness and context dependence of latent biosignal similarity",
        fontsize=13.2,
        fontweight="bold",
    )
    save_all(fig, out / "Figure_robustness_and_stimulus_sensitivity", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.results_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary_path = root / "three_dataset_independence_controlled_summary.csv"
    require_files([summary_path])
    summary = pd.read_csv(summary_path)
    configure_matplotlib()
    rng = np.random.default_rng(args.seed)

    make_figure_2(
        root,
        out,
        summary,
        rng,
        args.formats,
        args.dpi,
        args.ptb_display_max,
    )
    make_figure_3(root, out, summary, rng, args.formats, args.dpi)
    summary.to_csv(out / "reanalysis_primary_results.csv", index=False)

    print(f"Figures generated in: {out}")
    print("Figure 2 and Figure 3 were generated from numerical reanalysis outputs.")
    print("No generative image model was used.")


if __name__ == "__main__":
    main()
