from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import FEATURE_VERSION, basic_signal_features, centre_within_group, ensure_dir, robust_latent_embedding
from .reporting import plot_permutation, plot_unit_deltas, save_result, write_markdown_report
from .stats import benjamini_hochberg, compute_unit_deltas, encode_design, run_cluster_inference

PHYS_COLS = ["daqtime", "ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap", "video"]
ANNO_COLS = ["jstime", "valence", "arousal", "video"]

CASE_SOURCE_FS_HZ = 1000.0
CASE_ANALYSIS_FS_HZ = 1000.0
CASE_FEATURE_VERSION = FEATURE_VERSION + "_case_fullrate_1000hz_v3"


def _read_csv_flexible(path: Path, columns: List[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    current = [str(item).strip().lower() for item in frame.columns]
    if frame.shape[1] != len(columns) or not any(name in current for name in columns):
        frame = pd.read_csv(path, header=None)
        frame = frame.iloc[:, :len(columns)].copy()
        frame.columns = columns
    else:
        frame = frame.iloc[:, :len(columns)].copy()
        frame.columns = columns
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _subject_id(path: Path) -> str:
    match = re.search(r"sub[_-]?(\d+)", path.name.lower())
    return "sub_{:02d}".format(int(match.group(1))) if match else path.stem


def _find_files(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    physiological = []
    annotations = []
    for path in data_dir.rglob("*.csv"):
        text = str(path).lower()
        if not path.name.lower().startswith("sub"):
            continue
        if "physiological" in text:
            physiological.append(path)
        elif "annotation" in text:
            annotations.append(path)
    phys_interp = [p for p in physiological if "interpolated" in str(p).lower() and "non-interpolated" not in str(p).lower()]
    anno_interp = [p for p in annotations if "interpolated" in str(p).lower() and "non-interpolated" not in str(p).lower()]
    if phys_interp and anno_interp:
        physiological, annotations = phys_interp, anno_interp
    annotation_map = {_subject_id(path): path for path in annotations}
    return sorted([(sid, path, annotation_map[sid]) for path in physiological for sid in [_subject_id(path)] if sid in annotation_map])


def _bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    return "high" if value >= 5.0 else "low"


def _quadrant(valence: float, arousal: float) -> str:
    if not np.isfinite(valence) or not np.isfinite(arousal):
        return "missing"
    return "{}_{}".format("Vhigh" if valence >= 5.0 else "Vlow", "Ahigh" if arousal >= 5.0 else "Alow")


def _segment_features(segment: pd.DataFrame) -> np.ndarray:
    """Extract CASE features at the original common 1,000-Hz sampling rate.

    No segment-specific temporal subsampling is applied. This keeps the time
    scale of first-difference, autocorrelation, zero-crossing and spectral
    descriptors identical across all videos.
    """
    features = []
    for channel in ("ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"):
        values = segment[channel].to_numpy(dtype=float)
        features.append(basic_signal_features(values, fs=CASE_ANALYSIS_FS_HZ))
    return np.concatenate(features)


def _load_or_extract_features(data_dir: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ensure_dir(data_dir / "cache_independence_controlled")
    feature_path = cache_dir / "case_features_{}.npy".format(CASE_FEATURE_VERSION)
    meta_path = cache_dir / "case_meta_{}.csv".format(CASE_FEATURE_VERSION)
    if feature_path.exists() and meta_path.exists():
        X = np.load(feature_path)
        meta = pd.read_csv(meta_path)
        if len(meta) == X.shape[0]:
            print("Loaded CASE independence-controlled feature cache.")
            return X, meta
    pairs = _find_files(data_dir)
    if not pairs:
        raise FileNotFoundError("CASE physiological/annotation CSV pairs were not found under {}".format(data_dir))
    rows = []
    for subject, phys_path, anno_path in tqdm(pairs, desc="Extracting CASE subject-video features"):
        physiological = _read_csv_flexible(phys_path, PHYS_COLS)
        annotation = _read_csv_flexible(anno_path, ANNO_COLS)
        videos = sorted([value for value in pd.unique(physiological["video"]) if np.isfinite(value) and value > 0])
        for video in videos:
            segment = physiological[physiological["video"] == video]
            ratings = annotation[annotation["video"] == video]
            if len(segment) < 2000 or len(ratings) < 10:
                continue
            valence = float(np.nanmean(ratings["valence"]))
            arousal = float(np.nanmean(ratings["arousal"]))
            rows.append({
                "subject": subject,
                "video": int(video),
                "valence": valence,
                "arousal": arousal,
                "valence_bin": _bin(valence),
                "arousal_bin": _bin(arousal),
                "valence_arousal_quadrant": _quadrant(valence, arousal),
                "original_samples": int(len(segment)),
                "source_sampling_rate_hz": CASE_SOURCE_FS_HZ,
                "analysis_sampling_rate_hz": CASE_ANALYSIS_FS_HZ,
                "preprocessing_mode": "full_rate_no_subsampling",
                "features": _segment_features(segment),
            })
    if len(rows) < 50:
        raise RuntimeError("Too few CASE subject-video samples were extracted.")
    X = np.vstack([row["features"] for row in rows])
    meta = pd.DataFrame([{key: value for key, value in row.items() if key != "features"} for row in rows])
    np.save(feature_path, X)
    meta.to_csv(meta_path, index=False)
    return X, meta


def _leave_one_video_out(Z: np.ndarray, meta: pd.DataFrame, metric: str = "cosine") -> pd.DataFrame:
    rows = []
    for video in sorted(meta["video"].unique()):
        keep = meta["video"].to_numpy() != video
        design = encode_design(
            labels=meta.loc[keep, "valence_arousal_quadrant"].astype(str),
            units=meta.loc[keep, "subject"].astype(str),
            blocks=meta.loc[keep, "video"].astype(str),
        )
        _, unit_df, _ = compute_unit_deltas(Z[keep], design.labels, design.units, blocks=design.blocks, metric=metric)
        rows.append({"left_out_video": int(video), "mean_unit_delta": float(unit_df["delta"].mean()), "n_subjects": int(len(unit_df))})
    return pd.DataFrame(rows)


def run_case_independent(
    data_dir: Path,
    results_dir: Path,
    latent_dim: int = 8,
    n_permutations: int = 5000,
    n_bootstrap: int = 5000,
    sensitivity_permutations: int = 1000,
    sensitivity_bootstrap: int = 1000,
    seed: int = 20260730,
) -> Dict[str, object]:
    ensure_dir(data_dir)
    ensure_dir(results_dir)
    X, meta = _load_or_extract_features(data_dir)
    manifest_cols = [
        "subject", "video", "original_samples", "source_sampling_rate_hz",
        "analysis_sampling_rate_hz", "preprocessing_mode",
    ]
    if all(column in meta.columns for column in manifest_cols):
        meta[manifest_cols].to_csv(results_dir / "case_preprocessing_manifest.csv", index=False)
    Z, _, pca, _ = robust_latent_embedding(X, latent_dim=latent_dim)
    subjects = meta["subject"].astype(str).to_numpy()
    videos = meta["video"].astype(str).to_numpy()

    target_results = {}
    target_rows = []
    targets = ["valence_arousal_quadrant", "arousal_bin", "valence_bin"]
    for target_index, target in enumerate(targets):
        target_dir = ensure_dir(results_dir / ("primary_" + target if target_index == 0 else "secondary_" + target))
        result = run_cluster_inference(
            Z=Z,
            labels=meta[target].astype(str).to_numpy(),
            units=subjects,
            blocks=videos,
            metric="cosine",
            permutation_scheme="within_unit",
            n_permutations=n_permutations if target_index == 0 else sensitivity_permutations,
            n_bootstrap=n_bootstrap if target_index == 0 else sensitivity_bootstrap,
            seed=seed + target_index,
        )
        save_result(result, target_dir, "CASE", target)
        plot_unit_deltas(result, target_dir / "fig_subject_delta.png", "CASE: {} cross-subject contrast".format(target))
        plot_permutation(result, target_dir / "fig_permutation.png", "CASE: within-subject label permutation")
        row = dict(result["summary"])
        row.update({"target": target})
        target_rows.append(row)
        target_results[target] = result
    adjusted = benjamini_hochberg([row["permutation_p_two_sided"] for row in target_rows])
    for row, q_value in zip(target_rows, adjusted):
        row["fdr_q_two_sided"] = float(q_value)
    pd.DataFrame(target_rows).to_csv(results_dir / "target_summary_with_fdr.csv", index=False)

    primary = target_results["valence_arousal_quadrant"]
    pd.DataFrame(Z, columns=["z{}".format(i+1) for i in range(Z.shape[1])]).assign(
        subject=subjects, video=videos, quadrant=meta["valence_arousal_quadrant"].astype(str).to_numpy()
    ).to_csv(results_dir / "primary_valence_arousal_quadrant" / "latent_embedding.csv", index=False)

    # Remove the across-subject mean associated with each video and rerun.
    centred_dir = ensure_dir(results_dir / "sensitivity_video_centered")
    Z_centred = centre_within_group(Z, videos)
    centred = run_cluster_inference(
        Z=Z_centred,
        labels=meta["valence_arousal_quadrant"].astype(str).to_numpy(),
        units=subjects,
        blocks=videos,
        metric="cosine",
        permutation_scheme="within_unit",
        n_permutations=sensitivity_permutations,
        n_bootstrap=sensitivity_bootstrap,
        seed=seed + 20,
    )
    save_result(centred, centred_dir, "CASE", "Video-centred sensitivity")
    plot_unit_deltas(centred, centred_dir / "fig_subject_delta.png", "CASE: video-centred affective contrast")

    loo = _leave_one_video_out(Z, meta, metric="cosine")
    loo.to_csv(results_dir / "leave_one_video_out.csv", index=False)

    sensitivity_rows = []
    for dim in (4, 8, 16):
        Z_dim, _, _, _ = robust_latent_embedding(X, latent_dim=dim)
        for metric in ("cosine", "correlation", "neg_sqeuclidean"):
            result = run_cluster_inference(
                Z=Z_dim,
                labels=meta["valence_arousal_quadrant"].astype(str).to_numpy(),
                units=subjects,
                blocks=videos,
                metric=metric,
                permutation_scheme="within_unit",
                n_permutations=sensitivity_permutations,
                n_bootstrap=sensitivity_bootstrap,
                seed=seed + dim * 100 + len(metric),
            )
            row = dict(result["summary"])
            row.update({"latent_dim": dim, "dataset": "CASE"})
            sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "sensitivity_latent_dimension_and_metric.csv", index=False)

    primary_q = float(pd.DataFrame(target_rows).set_index("target").loc["valence_arousal_quadrant", "fdr_q_two_sided"])
    write_markdown_report(
        results_dir / "report.md",
        "CASE",
        "Cross-subject, cross-video affective affinity",
        "Physiological segments sharing the same valence-arousal quadrant are more similar across different participants and different videos than segments assigned to different quadrants.",
        primary,
        extra_lines=[
            "Both same-subject and same-video comparisons were excluded.",
            "Labels were permuted within each subject, preserving each participant's label distribution.",
            "The primary target was the valence-arousal quadrant; arousal and valence binary targets were FDR-adjusted.",
            "Primary two-sided FDR q = {:.6g}.".format(primary_q),
            "A video-centred sensitivity analysis and leave-one-video-out analysis were performed.",
        ],
    )
    summary = dict(primary["summary"])
    summary.update({
        "dataset": "CASE",
        "analysis_order": 2,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "primary_fdr_q_two_sided": primary_q,
        "video_centered_mean_delta": centred["summary"]["mean_unit_delta"],
        "video_centered_p_two_sided": centred["summary"]["permutation_p_two_sided"],
        "leave_one_video_out_min_delta": float(loo["mean_unit_delta"].min()),
        "leave_one_video_out_max_delta": float(loo["mean_unit_delta"].max()),
        "case_source_sampling_rate_hz": CASE_SOURCE_FS_HZ,
        "case_analysis_sampling_rate_hz": CASE_ANALYSIS_FS_HZ,
        "case_preprocessing_mode": "full_rate_no_subsampling",
        "case_feature_cache_version": CASE_FEATURE_VERSION,
    })
    return summary
