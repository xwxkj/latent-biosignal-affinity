from __future__ import annotations

import re
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .utils import (
    ensure_dir,
    basic_signal_features,
    make_latent,
    plot_embedding,
    plot_similarity_boxplot,
    plot_permutation,
    save_pairwise_summary,
    write_report,
)

CASE_URLS = [
    "https://springernature.figshare.com/ndownloader/files/16260497",
    "https://rmc.dlr.de/download/CASE_dataset/CASE_dataset.zip",
]

PHYS_COLS = [
    "daqtime", "ecg", "bvp", "gsr", "rsp", "skt",
    "emg_zygo", "emg_coru", "emg_trap", "video"
]

ANNO_COLS = ["jstime", "valence", "arousal", "video"]


def download_case(zip_path: Path):
    ensure_dir(zip_path.parent)
    if zip_path.exists() and zip_path.stat().st_size > 100_000_000:
        print(f"CASE zip already exists: {zip_path}")
        return

    last_err = None
    for url in CASE_URLS:
        try:
            print(f"Downloading CASE from: {url}")
            with urllib.request.urlopen(url, timeout=1800) as r, open(zip_path, "wb") as f:
                shutil.copyfileobj(r, f)
            if zip_path.stat().st_size > 100_000_000:
                print(f"Downloaded CASE zip: {zip_path}")
                return
        except Exception as e:
            last_err = e
            print(f"Warning: failed from {url}: {e}")

    raise RuntimeError(f"CASE download failed. Last error: {last_err}")


def extract_case(zip_path: Path, data_dir: Path):
    ensure_dir(data_dir)
    marker = data_dir / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    marker.write_text("ok")


def _read_csv_flexible(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If headers are numeric or malformed, reread without header.
    if df.shape[1] == len(expected_cols):
        current = [str(c).strip().lower() for c in df.columns]
        if not any(c in current for c in expected_cols):
            df = pd.read_csv(path, header=None)
            df.columns = expected_cols
        else:
            df.columns = [str(c).strip().lower() for c in df.columns]
    elif df.shape[1] >= len(expected_cols):
        df = df.iloc[:, :len(expected_cols)].copy()
        df.columns = expected_cols
    else:
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :len(expected_cols)].copy()
        df.columns = expected_cols

    for c in expected_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _subject_id(path: Path) -> str:
    m = re.search(r"sub[_-]?(\d+)", path.name.lower())
    return f"sub_{int(m.group(1)):02d}" if m else path.stem


def _find_case_files(data_dir: Path):
    """
    Robustly locate CASE physiological/annotation CSV pairs.

    Supported structures include:
    - .../interpolated/physiological/sub_XX.csv
    - .../interpolated/annotations/sub_XX.csv
    - .../physiological/sub_XX.csv
    - .../annotations/sub_XX.csv
    """
    phys_files = []
    anno_files = []

    for p in data_dir.rglob("*.csv"):
        s = str(p).lower()
        name = p.name.lower()

        if not name.startswith("sub"):
            continue

        if "physiological" in s:
            phys_files.append(p)
        elif "annotation" in s or "annotations" in s:
            anno_files.append(p)

    # Prefer interpolated files if both interpolated and non-interpolated versions exist.
    phys_interp = [
        p for p in phys_files
        if "interpolated" in str(p).lower() and "non-interpolated" not in str(p).lower()
    ]
    anno_interp = [
        p for p in anno_files
        if "interpolated" in str(p).lower() and "non-interpolated" not in str(p).lower()
    ]

    if phys_interp and anno_interp:
        phys_files, anno_files = phys_interp, anno_interp

    anno_map = {_subject_id(p): p for p in anno_files}
    pairs = []
    for pp in phys_files:
        sid = _subject_id(pp)
        if sid in anno_map:
            pairs.append((sid, pp, anno_map[sid]))

    return sorted(pairs, key=lambda x: x[0])

def _label_bin(x: float, threshold: float = 5.0) -> str:
    if not np.isfinite(x):
        return "missing"
    return "high" if x >= threshold else "low"


def _va_quadrant(valence: float, arousal: float) -> str:
    if not np.isfinite(valence) or not np.isfinite(arousal):
        return "missing"
    v = "Vhigh" if valence >= 5.0 else "Vlow"
    a = "Ahigh" if arousal >= 5.0 else "Alow"
    return f"{v}_{a}"


def _extract_features_for_segment(seg: pd.DataFrame) -> np.ndarray:
    chans = ["ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"]
    feats = []
    for c in chans:
        if c in seg.columns:
            x = seg[c].to_numpy(dtype=float)
            # Downsample if very long for speed, feature stability remains sufficient.
            if x.size > 120_000:
                step = max(1, x.size // 120_000)
                x = x[::step]
            feats.append(basic_signal_features(x, fs=1000.0))
    return np.concatenate(feats)


def _sample_pairs_excluding_groups(n: int, groups: np.ndarray, max_pairs: int, seed: int = 20260617) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = set()
    tries = 0
    while len(pairs) < max_pairs and tries < max_pairs * 100:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        tries += 1
        if i == j or groups[i] == groups[j]:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return np.asarray(list(pairs), dtype=int)


def _block_aware_pairwise_test(
    Z: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    max_pairs: int = 200000,
    n_permutations: int = 1000,
    seed: int = 20260617,
) -> Dict:
    labels = np.asarray(labels).astype(str)
    groups = np.asarray(groups).astype(str)
    valid = (labels != "missing") & pd.notnull(labels)

    Z = np.asarray(Z[valid], dtype=np.float64)
    labels = labels[valid]
    groups = groups[valid]

    if len(pd.unique(labels)) < 2:
        raise ValueError("Only one valid label class.")

    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    Zn = Z / norms

    n = len(labels)
    pairs = _sample_pairs_excluding_groups(n, groups, max_pairs=max_pairs, seed=seed)
    if pairs.size == 0:
        raise ValueError("No cross-subject pairs available.")

    sim_vals = np.sum(Zn[pairs[:, 0]] * Zn[pairs[:, 1]], axis=1)
    sim_vals = np.nan_to_num(sim_vals, nan=0.0, posinf=0.0, neginf=0.0)

    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    same_vals = sim_vals[same]
    diff_vals = sim_vals[~same]

    if same_vals.size < 10 or diff_vals.size < 10:
        raise ValueError("Too few same/different pairs.")

    observed = float(np.mean(same_vals) - np.mean(diff_vals))

    rng = np.random.default_rng(seed)
    null = np.zeros(n_permutations)
    for b in range(n_permutations):
        perm = rng.permutation(labels)
        same_perm = perm[pairs[:, 0]] == perm[pairs[:, 1]]
        if np.any(same_perm) and np.any(~same_perm):
            null[b] = float(np.mean(sim_vals[same_perm]) - np.mean(sim_vals[~same_perm]))
        else:
            null[b] = 0.0

    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    pooled = np.sqrt((np.var(same_vals) + np.var(diff_vals)) / 2.0 + 1e-12)
    cohend = float((np.mean(same_vals) - np.mean(diff_vals)) / pooled)

    try:
        _, mw_p = stats.mannwhitneyu(same_vals, diff_vals, alternative="greater")
        mw_p = float(mw_p)
    except Exception:
        mw_p = float("nan")

    return {
        "n_samples": int(n),
        "n_pairs": int(len(pairs)),
        "same_pairs": int(np.sum(same)),
        "different_pairs": int(np.sum(~same)),
        "mean_same": float(np.mean(same_vals)),
        "mean_different": float(np.mean(diff_vals)),
        "delta_same_minus_different": observed,
        "cohen_d": cohend,
        "permutation_p_value": p_value,
        "mannwhitney_p_value": mw_p,
        "same_values": same_vals,
        "different_values": diff_vals,
        "null_distribution": null,
    }


def run_case(
    data_dir: Path,
    zip_path: Path,
    results_dir: Path,
    download: bool,
    latent_dim: int,
    n_permutations: int,
    max_pairs: int,
):
    ensure_dir(data_dir)
    ensure_dir(results_dir)

    if download and not zip_path.exists():
        download_case(zip_path)

    if zip_path.exists():
        extract_case(zip_path, data_dir)

    pairs = _find_case_files(data_dir)
    if not pairs:
        raise FileNotFoundError(
            "CASE CSV files not found. Expected data/.../interpolated/physiological/sub_XX.csv "
            "and data/.../interpolated/annotations/sub_XX.csv."
        )

    rows = []
    failed = 0

    for sid, phys_path, anno_path in tqdm(pairs, desc="Extracting CASE features"):
        try:
            phys = _read_csv_flexible(phys_path, PHYS_COLS)
            anno = _read_csv_flexible(anno_path, ANNO_COLS)
        except Exception as e:
            print(f"Warning: failed reading {sid}: {e}")
            failed += 1
            continue

        videos = sorted([v for v in pd.unique(phys["video"]) if np.isfinite(v)])
        for vid in videos:
            if vid <= 0:
                continue

            seg = phys[phys["video"] == vid]
            ann = anno[anno["video"] == vid]

            if len(seg) < 2000 or len(ann) < 10:
                continue

            val = float(np.nanmean(ann["valence"].values))
            aro = float(np.nanmean(ann["arousal"].values))

            try:
                feat = _extract_features_for_segment(seg)
            except Exception:
                continue

            rows.append({
                "subject": sid,
                "video": int(vid),
                "valence": val,
                "arousal": aro,
                "valence_bin": _label_bin(val),
                "arousal_bin": _label_bin(aro),
                "valence_arousal_quadrant": _va_quadrant(val, aro),
                "features": feat,
            })

    if len(rows) < 50:
        raise RuntimeError(f"Too few CASE subject-video samples extracted: {len(rows)}")

    max_len = max(len(r["features"]) for r in rows)
    X = np.zeros((len(rows), max_len), dtype=float)
    for i, r in enumerate(rows):
        f = np.asarray(r["features"], dtype=float)
        X[i, :len(f)] = f

    meta = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in rows])
    Z, scaler, pca = make_latent(X, latent_dim=latent_dim)

    emb = pd.DataFrame(Z, columns=[f"z{k+1}" for k in range(Z.shape[1])])
    out = pd.concat([meta.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    out.to_csv(results_dir / "latent_embedding.csv", index=False)

    targets = ["valence_arousal_quadrant", "valence_bin", "arousal_bin"]
    summaries = []

    for target in targets:
        target_dir = results_dir / f"target_{target}"
        ensure_dir(target_dir)

        labels = meta[target].values
        groups = meta["subject"].values

        try:
            result = _block_aware_pairwise_test(
                Z,
                labels,
                groups,
                max_pairs=max_pairs,
                n_permutations=n_permutations,
            )
        except Exception as e:
            print(f"Skipping {target}: {e}")
            continue

        save_pairwise_summary(result, target_dir / "pairwise_summary.csv")
        plot_embedding(Z, labels, target_dir / "fig_embedding.png", f"CASE latent biosignal space: {target}")
        plot_similarity_boxplot(result, target_dir / "fig_similarity_boxplot.png", f"CASE affective affinity: {target}")
        plot_permutation(result, target_dir / "fig_permutation_test.png", f"CASE permutation test: {target}")

        write_report(
            target_dir / "report.md",
            dataset_name=f"CASE ({target})",
            hypothesis=f"Physiological latent biosignal similarity is higher for cross-subject video segments sharing the same {target} label than for segments with different labels.",
            result=result,
            extra={
                "target": target,
                "samples": int(len(meta)),
                "subjects": int(meta["subject"].nunique()),
                "failed_subjects": int(failed),
                "labels": sorted(pd.unique(labels).astype(str).tolist()),
                "same-subject_pairs_excluded": True,
                "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            },
        )

        summaries.append({
            "target": target,
            "n_samples": result["n_samples"],
            "n_pairs": result["n_pairs"],
            "same_pairs": result["same_pairs"],
            "different_pairs": result["different_pairs"],
            "mean_same": result["mean_same"],
            "mean_different": result["mean_different"],
            "delta": result["delta_same_minus_different"],
            "cohen_d": result["cohen_d"],
            "permutation_p": result["permutation_p_value"],
            "mannwhitney_p": result["mannwhitney_p_value"],
        })

    if not summaries:
        raise RuntimeError("No CASE target produced a valid pairwise test.")

    summary_df = pd.DataFrame(summaries).sort_values(["permutation_p", "delta"], ascending=[True, False])
    summary_df.to_csv(results_dir / "summary_all_targets.csv", index=False)

    primary = "valence_arousal_quadrant"
    primary_report = results_dir / f"target_{primary}" / "report.md"
    if primary_report.exists():
        (results_dir / "report.md").write_text(primary_report.read_text(), encoding="utf-8")
    else:
        top_target = summary_df.iloc[0]["target"]
        (results_dir / "report.md").write_text(
            (results_dir / f"target_{top_target}" / "report.md").read_text(),
            encoding="utf-8"
        )

    print("\nCASE analysis finished.")
    print(f"Samples: {len(meta)}, subjects: {meta['subject'].nunique()}")
    print(f"Summary: {results_dir / 'summary_all_targets.csv'}")
    print(f"Primary report: {results_dir / 'report.md'}")
