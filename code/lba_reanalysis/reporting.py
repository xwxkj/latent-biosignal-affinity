from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import ensure_dir


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


def save_result(result: Dict[str, object], out_dir: Path, dataset_name: str, analysis_name: str) -> None:
    ensure_dir(out_dir)
    summary = dict(result["summary"])
    summary["dataset"] = dataset_name
    summary["analysis"] = analysis_name
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    result["sample_deltas"].to_csv(out_dir / "sample_level_deltas.csv", index=False)
    result["unit_deltas"].to_csv(out_dir / "independent_unit_deltas.csv", index=False)
    pd.DataFrame({"permutation_null": result["permutation_null"]}).to_csv(out_dir / "permutation_null.csv", index=False)
    pd.DataFrame({"bootstrap_mean_delta": result["bootstrap_distribution"]}).to_csv(out_dir / "bootstrap_distribution.csv", index=False)


def plot_unit_deltas(result: Dict[str, object], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    values = result["unit_deltas"]["delta"].to_numpy(dtype=float)
    summary = result["summary"]
    rng = np.random.default_rng(20260730)
    jitter = rng.normal(0.0, 0.035, size=values.size)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(jitter, values, s=24, alpha=0.70)
    mean = summary["mean_unit_delta"]
    lo = summary["bootstrap_ci_95_low"]
    hi = summary["bootstrap_ci_95_high"]
    ax.errorbar([0], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", markersize=8, capsize=5, linewidth=1.8)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlim(-0.30, 0.30)
    ax.set_xticks([0])
    ax.set_xticklabels(["Independent units"])
    ax.set_ylabel("Unit-level Δ similarity")
    ax.set_title(title)
    ax.text(
        0.02, 0.98,
        "mean Δ = {:.4f}\n95% CI [{:.4f}, {:.4f}]\npermutation P(two-sided) = {:.4g}\nn = {}".format(
            mean, lo, hi, summary["permutation_p_two_sided"], summary["n_independent_units"]
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_permutation(result: Dict[str, object], out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    null = np.asarray(result["permutation_null"], dtype=float)
    observed = float(result["summary"]["mean_unit_delta"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.hist(null, bins=40, alpha=0.80)
    ax.axvline(observed, linestyle="--", linewidth=2.0)
    ax.set_xlabel("Mean unit-level Δ under the null")
    ax.set_ylabel("Permutation count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    out_path: Path,
    dataset_name: str,
    analysis_name: str,
    hypothesis: str,
    result: Dict[str, object],
    extra_lines: Optional[List[str]] = None,
) -> None:
    ensure_dir(out_path.parent)
    s = result["summary"]
    lines = [
        "# {} — {}".format(dataset_name, analysis_name),
        "",
        "## Hypothesis",
        "",
        hypothesis,
        "",
        "## Independence-controlled result",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        "| Independent units | {} |".format(s["n_independent_units"]),
        "| Samples | {} |".format(s["n_samples"]),
        "| Mean unit-level Δ | {:.6f} |".format(s["mean_unit_delta"]),
        "| Bootstrap 95% CI | [{:.6f}, {:.6f}] |".format(s["bootstrap_ci_95_low"], s["bootstrap_ci_95_high"]),
        "| Permutation P, two-sided | {:.6g} |".format(s["permutation_p_two_sided"]),
        "| Permutation P, directional | {:.6g} |".format(s["permutation_p_greater"]),
        "| Hedges g (unit-level Δ) | {:.4f} |".format(s["hedges_g"]),
        "| Wilcoxon P, two-sided | {:.6g} |".format(s["wilcoxon_p_two_sided"]),
        "| Descriptive mean similarity, same affinity | {:.6f} |".format(s["sample_weighted_mean_same"]),
        "| Descriptive mean similarity, different affinity | {:.6f} |".format(s["sample_weighted_mean_different"]),
        "",
        "The inferential sample size is the number of independent patients or participants, not the number of pairwise comparisons.",
    ]
    if extra_lines:
        lines.extend(["", "## Additional checks", ""] + extra_lines)
    lines.extend([
        "",
        "## Manuscript-ready template",
        "",
        "Across {n} independent units, the mean unit-level affinity contrast was Δ = {delta:.4f} "
        "(bootstrap 95% CI [{lo:.4f}, {hi:.4f}]; two-sided restricted-permutation P = {p:.4g}; Hedges g = {g:.3f}).".format(
            n=s["n_independent_units"], delta=s["mean_unit_delta"], lo=s["bootstrap_ci_95_low"],
            hi=s["bootstrap_ci_95_high"], p=s["permutation_p_two_sided"], g=s["hedges_g"]
        ),
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_combined_unit_deltas(summary_rows: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    labels = summary_rows["dataset"].tolist()
    means = summary_rows["mean_unit_delta"].to_numpy(dtype=float)
    lows = summary_rows["bootstrap_ci_95_low"].to_numpy(dtype=float)
    highs = summary_rows["bootstrap_ci_95_high"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(x, means, yerr=np.vstack([means-lows, highs-means]), fmt="o", markersize=8, capsize=5, linewidth=1.8)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean independent-unit Δ similarity")
    ax.set_title("Independence-controlled latent biosignal affinity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
