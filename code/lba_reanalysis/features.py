from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import shutil
import urllib.request

import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 20260730
FEATURE_VERSION = "independent_v1"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, path: Path, timeout: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print("Downloading: {}".format(url))
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)


def basic_signal_features(x: np.ndarray, fs: Optional[float] = None) -> np.ndarray:
    """Extract a compact, dataset-agnostic feature vector from one signal channel."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.zeros(24, dtype=np.float64)

    x = x - np.median(x)
    dx = np.diff(x)
    feats = [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.sqrt(np.mean(x ** 2))),
        float(np.mean(np.abs(x))),
        float(np.median(x)),
        float(np.percentile(x, 5)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 75)),
        float(np.percentile(x, 95)),
        float(stats.skew(x, bias=False)) if x.size > 8 else 0.0,
        float(stats.kurtosis(x, bias=False)) if x.size > 8 else 0.0,
        float(np.mean(dx)) if dx.size else 0.0,
        float(np.std(dx)) if dx.size else 0.0,
        float(np.sqrt(np.mean(dx ** 2))) if dx.size else 0.0,
        float(np.mean(np.abs(dx))) if dx.size else 0.0,
    ]

    hist, _ = np.histogram(x, bins=16, density=True)
    hist = hist + 1e-12
    feats.append(float(-np.sum(hist * np.log(hist))))
    feats.append(float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))))

    denom = float(np.sum(x ** 2) + 1e-12)
    for lag in (1, 2, 5):
        feats.append(float(np.sum(x[:-lag] * x[lag:]) / denom) if x.size > lag else 0.0)

    if fs is not None and x.size >= 16:
        try:
            freqs, pxx = signal.welch(x, fs=float(fs), nperseg=min(256, x.size))
            total = float(np.trapz(pxx, freqs) + 1e-12)
            for lo, hi in ((0.05, 0.5), (0.5, 5.0), (5.0, 15.0), (15.0, 40.0)):
                mask = (freqs >= lo) & (freqs < hi)
                value = float(np.trapz(pxx[mask], freqs[mask]) / total) if np.any(mask) else 0.0
                feats.append(value)
        except Exception:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    out = np.asarray(feats, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def robust_latent_embedding(
    X: np.ndarray,
    latent_dim: int = 8,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> Tuple[np.ndarray, StandardScaler, PCA, Dict[str, np.ndarray]]:
    """Create a numerically stable PCA embedding with explicit percentile capping."""
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    lower = np.percentile(X, lower_percentile, axis=0)
    upper = np.percentile(X, upper_percentile, axis=0)
    invalid = (~np.isfinite(lower)) | (~np.isfinite(upper)) | (upper <= lower)
    if np.any(invalid):
        med = np.median(X[:, invalid], axis=0)
        lower[invalid] = med - 0.5
        upper[invalid] = med + 0.5
    Xc = np.clip(X, lower, upper)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip(Xs, -8.0, 8.0)

    n_components = int(max(2, min(latent_dim, Xs.shape[0] - 1, Xs.shape[1])))
    pca = PCA(n_components=n_components, svd_solver="full", random_state=SEED)
    Z = pca.fit_transform(Xs)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    preprocessing = {"lower": lower, "upper": upper, "standardized_features": Xs}
    return Z, scaler, pca, preprocessing


def residualize_matrix(Y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Remove linear effects of covariates from each column of Y."""
    Y = np.asarray(Y, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
    if design.ndim == 1:
        design = design[:, None]
    Xd = np.column_stack([np.ones(design.shape[0]), design])
    beta, _, _, _ = np.linalg.lstsq(Xd, Y, rcond=None)
    residual = Y - Xd.dot(beta)
    return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)


def centre_within_group(Y: np.ndarray, groups: Sequence[object]) -> np.ndarray:
    """Subtract the group-specific column mean from each row."""
    Y = np.asarray(Y, dtype=np.float64)
    groups_arr = np.asarray(groups).astype(str)
    out = Y.copy()
    for group in np.unique(groups_arr):
        idx = np.flatnonzero(groups_arr == group)
        out[idx] = out[idx] - np.mean(out[idx], axis=0, keepdims=True)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
