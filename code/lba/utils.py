from __future__ import annotations

import ast
import math
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


RNG = np.random.default_rng(20260617)


def download_file(url: str, path: Path, timeout: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def basic_signal_features(x: np.ndarray, fs: float | None = None) -> np.ndarray:
    """
    Robust feature extractor for one 1D physiological signal.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.zeros(22, dtype=float)

    x = x - np.nanmedian(x)
    dx = np.diff(x)

    feats = [
        np.mean(x),
        np.std(x),
        np.sqrt(np.mean(x ** 2)),
        np.mean(np.abs(x)),
        np.median(x),
        np.percentile(x, 5),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.percentile(x, 95),
        stats.skew(x, bias=False) if x.size > 8 else 0.0,
        stats.kurtosis(x, bias=False) if x.size > 8 else 0.0,
        np.mean(dx) if dx.size else 0.0,
        np.std(dx) if dx.size else 0.0,
        np.sqrt(np.mean(dx ** 2)) if dx.size else 0.0,
        np.mean(np.abs(dx)) if dx.size else 0.0,
    ]

    # Entropy-like histogram feature
    hist, _ = np.histogram(x, bins=16, density=True)
    hist = hist + 1e-12
    feats.append(float(-np.sum(hist * np.log(hist))))

    # Zero crossing rate
    feats.append(float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if x.size > 1 else 0.0)

    # Autocorrelation lags
    denom = np.sum(x ** 2) + 1e-12
    for lag in [1, 2, 5]:
        if x.size > lag:
            feats.append(float(np.sum(x[:-lag] * x[lag:]) / denom))
        else:
            feats.append(0.0)

    # Frequency features
    if fs is not None and x.size >= 16:
        try:
            f, pxx = signal.welch(x, fs=fs, nperseg=min(256, x.size))
            total = np.trapz(pxx, f) + 1e-12
            for lo, hi in [(0.05, 0.5), (0.5, 5.0), (5.0, 15.0), (15.0, 40.0)]:
                mask = (f >= lo) & (f < hi)
                feats.append(float(np.trapz(pxx[mask], f[mask]) / total) if np.any(mask) else 0.0)
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    arr = np.asarray(feats, dtype=float)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def make_latent(X: np.ndarray, latent_dim: int = 8) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """
    Numerically stable latent representation.

    Key safeguards:
    1. Replace NaN/Inf values.
    2. Winsorize each feature column to suppress extreme artifacts.
    3. Standardize features.
    4. Clip standardized values before PCA.
    5. Use full SVD PCA rather than covariance-eigendecomposition.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Column-wise robust winsorization.
    lo = np.nanpercentile(X, 0.5, axis=0)
    hi = np.nanpercentile(X, 99.5, axis=0)

    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[bad] = np.nanmedian(X[:, bad], axis=0) if np.any(bad) else lo[bad]
    hi[bad] = lo[bad] + 1.0 if np.any(bad) else hi[bad]

    X = np.minimum(np.maximum(X, lo), hi)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    # Prevent rare extreme standardized values from destabilizing PCA.
    Xs = np.clip(Xs, -8.0, 8.0)

    n_comp = int(max(2, min(latent_dim, Xs.shape[0] - 1, Xs.shape[1])))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=20260617)
    Z = pca.fit_transform(Xs)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, scaler, pca

def sample_pairs(n: int, max_pairs: int = 200000, seed: int = 20260617) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return np.asarray(pairs, dtype=int)
    pairs = set()
    while len(pairs) < max_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    return np.asarray(list(pairs), dtype=int)


def pairwise_affinity_test(
    Z: np.ndarray,
    labels: np.ndarray,
    max_pairs: int = 200000,
    n_permutations: int = 1000,
    seed: int = 20260617,
) -> Dict:
    """
    Numerically stable pairwise affinity test.

    This version avoids constructing the full N x N cosine-similarity matrix.
    It samples pairs first, L2-normalizes latent representations, and computes
    pairwise cosine values only for the sampled pairs.
    """
    labels = np.asarray(labels)
    valid = pd.notnull(labels)
    Z = np.asarray(Z[valid], dtype=np.float64)
    labels = labels[valid]

    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.clip(Z, -1e6, 1e6)

    n = len(labels)
    if n < 5:
        raise ValueError("Too few valid samples for pairwise test.")

    # Stable L2 normalization.
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Zn = Z / norms
    Zn = np.nan_to_num(Zn, nan=0.0, posinf=0.0, neginf=0.0)

    pairs = sample_pairs(n, max_pairs=max_pairs, seed=seed)

    # Compute cosine similarity only for sampled pairs.
    sim_vals = np.sum(Zn[pairs[:, 0]] * Zn[pairs[:, 1]], axis=1)
    sim_vals = np.nan_to_num(sim_vals, nan=0.0, posinf=0.0, neginf=0.0)

    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]

    same_vals = sim_vals[same]
    diff_vals = sim_vals[~same]

    if same_vals.size == 0 or diff_vals.size == 0:
        raise ValueError("Insufficient same-affinity or different-affinity pairs.")

    observed = float(np.mean(same_vals) - np.mean(diff_vals))

    rng = np.random.default_rng(seed)
    null = np.zeros(n_permutations, dtype=float)

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
        u_stat, mw_p = stats.mannwhitneyu(same_vals, diff_vals, alternative="greater")
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

def plot_embedding(Z: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    if Z.shape[1] < 2:
        z2 = np.c_[Z[:, 0], np.zeros(Z.shape[0])]
    else:
        z2 = Z[:, :2]
    plt.figure(figsize=(7.5, 5.5))
    cats = pd.Series(labels).astype(str).fillna("NA").values
    uniq = sorted(pd.unique(cats))
    for u in uniq:
        idx = cats == u
        plt.scatter(z2[idx, 0], z2[idx, 1], s=18, alpha=0.75, label=u)
    plt.title(title)
    plt.xlabel("Latent component 1")
    plt.ylabel("Latent component 2")
    plt.legend(frameon=False, fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_similarity_boxplot(result: Dict, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    same = result["same_values"]
    diff = result["different_values"]
    # For plotting, downsample to keep the image light
    rng = np.random.default_rng(20260617)
    if same.size > 5000:
        same_plot = rng.choice(same, 5000, replace=False)
    else:
        same_plot = same
    if diff.size > 5000:
        diff_plot = rng.choice(diff, 5000, replace=False)
    else:
        diff_plot = diff
    plt.figure(figsize=(6.5, 5.0))
    plt.boxplot([same_plot, diff_plot], labels=["same affinity", "different affinity"], showfliers=False)
    plt.ylabel("Cosine similarity in latent biosignal space")
    plt.title(title)
    txt = f"Δ={result['delta_same_minus_different']:.4f}, p_perm={result['permutation_p_value']:.3g}"
    plt.text(0.5, 0.03, txt, transform=plt.gca().transAxes, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_permutation(result: Dict, out_path: Path, title: str) -> None:
    ensure_dir(out_path.parent)
    null = result["null_distribution"]
    obs = result["delta_same_minus_different"]
    plt.figure(figsize=(6.5, 5.0))
    plt.hist(null, bins=40, alpha=0.8)
    plt.axvline(obs, linestyle="--", linewidth=2)
    plt.xlabel("Null Δ similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_report(
    out_path: Path,
    dataset_name: str,
    hypothesis: str,
    result: Dict,
    extra: Dict | None = None,
) -> None:
    ensure_dir(out_path.parent)
    extra = extra or {}
    interpretation = (
        "supports the latent biosignal affinity hypothesis"
        if result["permutation_p_value"] < 0.05 and result["delta_same_minus_different"] > 0
        else "does not provide statistically significant support under this configuration"
    )
    md = f"""# {dataset_name} latent biosignal affinity report

## Hypothesis

{hypothesis}

## Pairwise similarity result

| Metric | Value |
|---|---:|
| Samples | {result['n_samples']} |
| Sampled pairs | {result['n_pairs']} |
| Same-affinity pairs | {result['same_pairs']} |
| Different-affinity pairs | {result['different_pairs']} |
| Mean similarity, same affinity | {result['mean_same']:.6f} |
| Mean similarity, different affinity | {result['mean_different']:.6f} |
| Delta, same minus different | {result['delta_same_minus_different']:.6f} |
| Cohen's d | {result['cohen_d']:.4f} |
| Permutation p-value | {result['permutation_p_value']:.6g} |
| Mann-Whitney p-value | {result['mannwhitney_p_value']:.6g} |

## Interpretation

Under this configuration, the result **{interpretation}**.

A positive and significant delta means that samples sharing the same target affinity label are closer in the latent biosignal space than samples with different labels.

## Extra information

```json
{json_dumps(extra)}
```

## Manuscript-ready sentence

The latent biosignal similarity was higher for same-affinity pairs than for different-affinity pairs (Δ = {result['delta_same_minus_different']:.4f}, permutation p = {result['permutation_p_value']:.3g}), indicating that the selected biosignal modality contains a statistically detectable affinity-related structure.

"""
    out_path.write_text(md, encoding="utf-8")


def json_dumps(obj):
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def save_pairwise_summary(result: Dict, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    pd.DataFrame([{
        k: v for k, v in result.items()
        if k not in ["same_values", "different_values", "null_distribution"]
    }]).to_csv(out_path, index=False)
