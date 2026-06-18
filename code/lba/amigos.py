from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
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

LABEL_NAMES = [
    "arousal", "valence", "dominance", "liking", "familiarity",
    "neutral", "disgust", "happiness", "surprise", "anger",
    "fear", "sadness"
]

SKIPPED_SUBJECTS = {9, 12, 21, 22, 23, 24, 33}


def _binary_rating(x: float, threshold: float = 5.0) -> str:
    if not np.isfinite(x):
        return "missing"
    return "high" if x >= threshold else "low"


def _valence_arousal_quadrant(valence: float, arousal: float) -> str:
    if not np.isfinite(valence) or not np.isfinite(arousal):
        return "missing"
    v = "Vhigh" if valence >= 5.0 else "Vlow"
    a = "Ahigh" if arousal >= 5.0 else "Alow"
    return f"{v}_{a}"


def _emotion_label(rating: np.ndarray) -> str:
    """
    Pick strongest basic emotion among neutral/disgust/happiness/surprise/anger/fear/sadness.
    If all are zero or missing, return missing.
    """
    if rating.size < 12:
        return "missing"
    basic = rating[5:12]
    if not np.all(np.isfinite(basic)):
        return "missing"
    if np.max(basic) <= 0:
        return "missing"
    return LABEL_NAMES[5 + int(np.argmax(basic))]


def _extract_subject_id(path: Path) -> int:
    m = re.search(r"Data_Preprocessed_P(\d+)\.mat", path.name)
    if not m:
        return -1
    return int(m.group(1))


def _safe_rating(labels, trial_id: int) -> np.ndarray:
    try:
        r = labels[trial_id]
        r = np.asarray(r).squeeze()
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.size >= 12:
            return r[:12]
    except Exception:
        pass
    return np.full(12, np.nan)


def _safe_trial(samples, trial_id: int):
    try:
        x = samples[trial_id]
        x = np.asarray(x, dtype=float)
        if x.size == 0 or x.ndim != 2:
            return None
        return x
    except Exception:
        return None


def _trial_to_channel_time(x: np.ndarray) -> np.ndarray:
    """
    AMIGOS joined_data trials are commonly timestep x channel.
    Convert to channel x timestep.
    """
    if x.shape[0] <= 32 and x.shape[1] > x.shape[0]:
        return x
    if x.shape[1] <= 32 and x.shape[0] > x.shape[1]:
        return x.T
    # fallback: assume smaller dimension is channel
    if x.shape[0] <= x.shape[1]:
        return x
    return x.T


def _extract_trial_features(x_channel_time: np.ndarray) -> np.ndarray:
    # Use 17 channels when available: 14 EEG + 2 ECG + 1 GSR.
    n_channels = min(17, x_channel_time.shape[0])
    x = x_channel_time[:n_channels, :]

    # TorchEEG convention uses first 640 samples as baseline.
    start = 640 if x.shape[1] > 1024 else 0
    dynamic = x[:, start:]
    baseline = x[:, :start] if start > 0 else None

    feats = []
    for ch in range(n_channels):
        sig = dynamic[ch]
        if baseline is not None and baseline.shape[1] > 0:
            sig = sig - np.nanmean(baseline[ch])
        feats.append(basic_signal_features(sig, fs=128.0))

    return np.concatenate(feats)


def _sample_pairs_excluding_groups(n: int, groups: np.ndarray, max_pairs: int, seed: int = 20260617) -> np.ndarray:
    total = n * (n - 1) // 2
    if total <= max_pairs * 2:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if groups[i] != groups[j]:
                    pairs.append((i, j))
        if len(pairs) > max_pairs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[k] for k in idx]
        return np.asarray(pairs, dtype=int)

    rng = np.random.default_rng(seed)
    pairs = set()
    tries = 0
    while len(pairs) < max_pairs and tries < max_pairs * 50:
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

    # Require at least two labels.
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
        raise ValueError("Too few same or different pairs.")

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


def run_amigos(
    data_dir: Path,
    results_dir: Path,
    latent_dim: int,
    n_permutations: int,
    max_pairs: int,
):
    ensure_dir(results_dir)

    mat_files = sorted(data_dir.glob("Data_Preprocessed_P*.mat"))
    if not mat_files:
        raise FileNotFoundError(
            "AMIGOS .mat files not found. Put Data_Preprocessed_P01.mat ... Data_Preprocessed_P40.mat in data/amigos/data_preprocessed/."
        )

    rows = []
    failed = 0

    for mat_path in tqdm(mat_files, desc="Extracting AMIGOS trial features"):
        subject = _extract_subject_id(mat_path)
        if subject in SKIPPED_SUBJECTS:
            continue

        try:
            data = loadmat(mat_path, verify_compressed_data_integrity=False)
            samples = data["joined_data"][0]
            labels = data["labels_selfassessment"][0]
        except Exception as e:
            print(f"Warning: failed loading {mat_path.name}: {e}")
            failed += 1
            continue

        n_trials = min(len(samples), len(labels), 20)
        for trial_id in range(n_trials):
            x = _safe_trial(samples, trial_id)
            rating = _safe_rating(labels, trial_id)

            if x is None:
                continue

            try:
                xct = _trial_to_channel_time(x)
                feat = _extract_trial_features(xct)
            except Exception:
                continue

            row = {
                "subject": f"P{subject:02d}",
                "trial": trial_id + 1,
                "features": feat,
            }

            for k, name in enumerate(LABEL_NAMES):
                row[name] = float(rating[k]) if k < rating.size else np.nan

            row["valence_bin"] = _binary_rating(row["valence"])
            row["arousal_bin"] = _binary_rating(row["arousal"])
            row["dominance_bin"] = _binary_rating(row["dominance"])
            row["liking_bin"] = _binary_rating(row["liking"])
            row["familiarity_bin"] = _binary_rating(row["familiarity"])
            row["valence_arousal_quadrant"] = _valence_arousal_quadrant(row["valence"], row["arousal"])
            row["emotion_label"] = _emotion_label(rating)

            rows.append(row)

    if len(rows) < 50:
        raise RuntimeError(f"Too few AMIGOS samples extracted: {len(rows)}.")

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

    targets = [
        "valence_arousal_quadrant",
        "valence_bin",
        "arousal_bin",
        "dominance_bin",
        "liking_bin",
        "familiarity_bin",
        "emotion_label",
    ]

    summaries = []
    best = None

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
        plot_embedding(Z, labels, target_dir / "fig_embedding.png", f"AMIGOS latent biosignal space: {target}")
        plot_similarity_boxplot(result, target_dir / "fig_similarity_boxplot.png", f"AMIGOS affinity: {target}")
        plot_permutation(result, target_dir / "fig_permutation_test.png", f"AMIGOS permutation test: {target}")

        write_report(
            target_dir / "report.md",
            dataset_name=f"AMIGOS ({target})",
            hypothesis=f"Multimodal biosignal similarity is higher for cross-subject trial pairs sharing the same {target} label than for pairs with different labels.",
            result=result,
            extra={
                "target": target,
                "samples": int(len(meta)),
                "subjects": int(meta["subject"].nunique()),
                "failed_files": int(failed),
                "labels": sorted(pd.unique(labels).astype(str).tolist()),
                "same-subject_pairs_excluded": True,
                "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            },
        )

        rec = {
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
        }
        summaries.append(rec)

        if best is None or (rec["delta"] > best["delta"] and rec["permutation_p"] <= 0.05):
            best = rec

    if not summaries:
        raise RuntimeError("No AMIGOS target produced a valid pairwise test.")

    summary_df = pd.DataFrame(summaries).sort_values(["permutation_p", "delta"], ascending=[True, False])
    summary_df.to_csv(results_dir / "summary_all_targets.csv", index=False)

    primary = "valence_arousal_quadrant"
    primary_report = results_dir / f"target_{primary}" / "report.md"
    if primary_report.exists():
        (results_dir / "report.md").write_text(primary_report.read_text(), encoding="utf-8")
    else:
        top_target = summary_df.iloc[0]["target"]
        top_report = results_dir / f"target_{top_target}" / "report.md"
        (results_dir / "report.md").write_text(top_report.read_text(), encoding="utf-8")

    print(f"\nAMIGOS analysis finished.")
    print(f"Samples: {len(meta)}, subjects: {meta['subject'].nunique()}")
    print(f"Summary: {results_dir / 'summary_all_targets.csv'}")
    print(f"Primary report: {results_dir / 'report.md'}")
