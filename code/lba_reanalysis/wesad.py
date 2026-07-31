from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import FEATURE_VERSION, basic_signal_features, download_file, ensure_dir, robust_latent_embedding
from .reporting import plot_permutation, plot_unit_deltas, save_result, write_markdown_report
from .stats import run_cluster_inference

WESAD_DOWNLOAD = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
STATE_MAP = {1: "baseline", 2: "stress", 3: "amusement"}
CHEST_FS = {"ACC": 700.0, "ECG": 700.0, "EDA": 700.0, "EMG": 700.0, "Resp": 700.0, "Temp": 700.0}
WRIST_FS = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}


def _extract_zip(zip_path: Path, data_dir: Path) -> None:
    ensure_dir(data_dir)
    marker = data_dir / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
    marker.write_text("ok", encoding="utf-8")


def _contiguous_intervals(indices: np.ndarray) -> List[Tuple[int, int]]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    return [(int(indices[s]), int(indices[e]) + 1) for s, e in zip(starts, ends)]


def _slice_by_intervals(array: np.ndarray, intervals_700hz: List[Tuple[int, int]], target_fs: float) -> np.ndarray:
    array = np.asarray(array)
    pieces = []
    for start, end in intervals_700hz:
        target_start = int(np.floor(start / 700.0 * target_fs))
        target_end = int(np.ceil(end / 700.0 * target_fs))
        target_start = max(0, min(target_start, array.shape[0]))
        target_end = max(target_start, min(target_end, array.shape[0]))
        if target_end > target_start:
            pieces.append(array[target_start:target_end])
    if not pieces:
        if array.ndim == 1:
            return np.empty(0)
        return np.empty((0, array.shape[1]))
    return np.concatenate(pieces, axis=0)


def _channel_feature_centroid(array: np.ndarray, fs: float, n_windows: int = 5) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    channel_features = []
    for channel in range(array.shape[1]):
        signal_values = array[:, channel]
        windows = [chunk for chunk in np.array_split(signal_values, n_windows) if chunk.size >= max(8, int(fs))]
        if not windows:
            channel_features.append(basic_signal_features(signal_values, fs=fs))
        else:
            channel_features.append(np.mean(np.vstack([basic_signal_features(chunk, fs=fs) for chunk in windows]), axis=0))
    return np.concatenate(channel_features)


def _extract_subject_state_rows(pickle_path: Path) -> List[Dict[str, object]]:
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    subject = pickle_path.stem
    labels = np.asarray(data["label"]).reshape(-1)
    signals = data["signal"]
    rows = []
    for state_id, state_name in STATE_MAP.items():
        indices = np.flatnonzero(labels == state_id)
        intervals = _contiguous_intervals(indices)
        if not intervals:
            continue
        features = []
        chest = signals.get("chest", {})
        wrist = signals.get("wrist", {})
        for name in ("ACC", "ECG", "EDA", "EMG", "Resp", "Temp"):
            if name in chest:
                segment = _slice_by_intervals(chest[name], intervals, CHEST_FS[name])
                if segment.shape[0] > 0:
                    features.append(_channel_feature_centroid(segment, CHEST_FS[name]))
        for name in ("ACC", "BVP", "EDA", "TEMP"):
            if name in wrist:
                segment = _slice_by_intervals(wrist[name], intervals, WRIST_FS[name])
                if segment.shape[0] > 0:
                    features.append(_channel_feature_centroid(segment, WRIST_FS[name]))
        if features:
            rows.append({"subject": subject, "state": state_name, "features": np.concatenate(features)})
    return rows


def _load_or_extract_features(data_dir: Path, zip_path: Path, download: bool) -> Tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ensure_dir(data_dir / "cache_independence_controlled")
    feature_path = cache_dir / "wesad_chest_wrist_features_{}.npy".format(FEATURE_VERSION)
    meta_path = cache_dir / "wesad_chest_wrist_meta_{}.csv".format(FEATURE_VERSION)
    if feature_path.exists() and meta_path.exists():
        X = np.load(feature_path)
        meta = pd.read_csv(meta_path)
        if len(meta) == X.shape[0]:
            print("Loaded WESAD independence-controlled feature cache.")
            return X, meta
    if download and not zip_path.exists():
        download_file(WESAD_DOWNLOAD, zip_path, timeout=1800)
    if zip_path.exists():
        _extract_zip(zip_path, data_dir)
    pickle_files = sorted(data_dir.rglob("S*.pkl"))
    if not pickle_files:
        raise FileNotFoundError("No WESAD subject pickle files found under {}".format(data_dir))
    rows = []
    for path in tqdm(pickle_files, desc="Extracting WESAD chest-and-wrist features"):
        rows.extend(_extract_subject_state_rows(path))
    if len(rows) < 30:
        raise RuntimeError("Too few WESAD subject-state representations were extracted.")
    X = np.vstack([row["features"] for row in rows])
    meta = pd.DataFrame([{key: value for key, value in row.items() if key != "features"} for row in rows])
    np.save(feature_path, X)
    meta.to_csv(meta_path, index=False)
    return X, meta


def run_wesad_independent(
    data_dir: Path,
    zip_path: Path,
    results_dir: Path,
    download: bool = False,
    latent_dim: int = 8,
    n_permutations: int = 5000,
    n_bootstrap: int = 5000,
    sensitivity_permutations: int = 1000,
    sensitivity_bootstrap: int = 1000,
    seed: int = 20260730,
) -> Dict[str, object]:
    ensure_dir(data_dir)
    ensure_dir(results_dir)
    X, meta = _load_or_extract_features(data_dir, zip_path, download=download)
    Z, _, pca, _ = robust_latent_embedding(X, latent_dim=latent_dim)
    subjects = meta["subject"].astype(str).to_numpy()
    states = meta["state"].astype(str).to_numpy()

    primary_dir = ensure_dir(results_dir / "primary_cross_subject")
    primary = run_cluster_inference(
        Z=Z,
        labels=states,
        units=subjects,
        metric="cosine",
        permutation_scheme="within_unit",
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    save_result(primary, primary_dir, "WESAD", "Cross-subject repeated-measures analysis")
    plot_unit_deltas(primary, primary_dir / "fig_subject_delta.png", "WESAD: subject-level stress/affect contrast")
    plot_permutation(primary, primary_dir / "fig_permutation.png", "WESAD: within-subject state permutation")
    sample_with_meta = primary["sample_deltas"].copy()
    sample_with_meta = sample_with_meta.merge(meta.reset_index().rename(columns={"index": "sample_index"}), on="sample_index", how="left")
    sample_with_meta.to_csv(primary_dir / "sample_deltas_with_state.csv", index=False)
    pd.DataFrame(Z, columns=["z{}".format(i+1) for i in range(Z.shape[1])]).assign(subject=subjects, state=states).to_csv(
        primary_dir / "latent_embedding.csv", index=False
    )

    state_summary = sample_with_meta.groupby("state", as_index=False).agg(
        n_subject_state_samples=("delta", "size"), mean_sample_delta=("delta", "mean"), median_sample_delta=("delta", "median")
    )
    state_summary.to_csv(results_dir / "state_specific_deltas.csv", index=False)

    sensitivity_rows = []
    for dim in (4, 8, 16):
        Z_dim, _, _, _ = robust_latent_embedding(X, latent_dim=dim)
        for metric in ("cosine", "correlation", "neg_sqeuclidean"):
            result = run_cluster_inference(
                Z=Z_dim,
                labels=states,
                units=subjects,
                metric=metric,
                permutation_scheme="within_unit",
                n_permutations=sensitivity_permutations,
                n_bootstrap=sensitivity_bootstrap,
                seed=seed + dim * 100 + len(metric),
            )
            row = dict(result["summary"])
            row.update({"latent_dim": dim, "dataset": "WESAD"})
            sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "sensitivity_latent_dimension_and_metric.csv", index=False)

    write_markdown_report(
        results_dir / "report.md",
        "WESAD",
        "Cross-subject stress/affect affinity",
        "Participants show more similar chest-and-wrist physiological representations when they are in the same experimentally induced state than when they are in different states.",
        primary,
        extra_lines=[
            "All same-subject comparisons were excluded.",
            "State labels were permuted within each subject, preserving the repeated-measures design.",
            "Bootstrap confidence intervals used subjects, not pairwise comparisons, as the resampling unit.",
            "Features included both chest and wrist modalities.",
        ],
    )
    summary = dict(primary["summary"])
    summary.update({"dataset": "WESAD", "analysis_order": 1, "pca_explained_variance": pca.explained_variance_ratio_.tolist()})
    return summary
