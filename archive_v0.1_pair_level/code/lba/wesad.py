from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    ensure_dir,
    download_file,
    basic_signal_features,
    make_latent,
    pairwise_affinity_test,
    plot_embedding,
    plot_similarity_boxplot,
    plot_permutation,
    write_report,
    save_pairwise_summary,
)

WESAD_DOWNLOAD = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
STATE_MAP = {
    1: "baseline",
    2: "stress",
    3: "amusement",
}


def download_wesad(zip_path: Path):
    ensure_dir(zip_path.parent)
    download_file(WESAD_DOWNLOAD, zip_path, timeout=1800)


def extract_wesad(zip_path: Path, data_dir: Path):
    ensure_dir(data_dir)
    marker = data_dir / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    marker.write_text("ok")


def find_subject_pickles(data_dir: Path):
    return sorted(data_dir.rglob("S*.pkl"))


def safe_get_signal(dct, keys):
    cur = dct
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def extract_wesad_subject_features(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    sid = pkl_path.stem
    labels = np.asarray(data.get("label"))
    signal = data.get("signal", {})

    # Use chest signals because they share 700 Hz sampling and include ECG/EDA/EMG/Resp/Temp/ACC.
    chest = signal.get("chest", {})
    fs = 700.0

    rows = []
    for state_id, state_name in STATE_MAP.items():
        idx = np.where(labels == state_id)[0]
        if idx.size < 700:
            continue

        # To keep computation light, use up to 5 non-overlapping windows per subject/state.
        n_windows = 5
        splits = np.array_split(idx, n_windows)
        state_features = []
        for seg in splits:
            if seg.size < 700:
                continue
            feats = []
            for name, arr in chest.items():
                x = np.asarray(arr)
                if x.ndim == 1:
                    feats.append(basic_signal_features(x[seg], fs=fs))
                elif x.ndim == 2:
                    for ch in range(x.shape[1]):
                        feats.append(basic_signal_features(x[seg, ch], fs=fs))
            if feats:
                state_features.append(np.concatenate(feats))
        if state_features:
            # Use centroid per subject-state.
            rows.append({
                "subject": sid,
                "state": state_name,
                "features": np.mean(np.vstack(state_features), axis=0),
            })
    return rows


def run_wesad(
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
        download_wesad(zip_path)

    if zip_path.exists():
        extract_wesad(zip_path, data_dir)

    pkl_files = find_subject_pickles(data_dir)
    if not pkl_files:
        raise FileNotFoundError(
            "No WESAD subject pickle files found. Run with --download or place WESAD.zip at data/WESAD.zip."
        )

    rows = []
    for p in tqdm(pkl_files, desc="Extracting WESAD features"):
        try:
            rows.extend(extract_wesad_subject_features(p))
        except Exception as e:
            print(f"Warning: failed to process {p}: {e}")

    if len(rows) < 10:
        raise RuntimeError("Too few WESAD subject-state samples extracted.")

    X = np.vstack([r["features"] for r in rows])
    labels = np.asarray([r["state"] for r in rows])
    subjects = np.asarray([r["subject"] for r in rows])

    Z, scaler, pca = make_latent(X, latent_dim=latent_dim)
    result = pairwise_affinity_test(
        Z, labels,
        max_pairs=max_pairs,
        n_permutations=n_permutations,
    )

    emb = pd.DataFrame(Z, columns=[f"z{k+1}" for k in range(Z.shape[1])])
    emb.insert(0, "subject", subjects)
    emb["state"] = labels
    emb.to_csv(results_dir / "latent_embedding.csv", index=False)

    save_pairwise_summary(result, results_dir / "pairwise_summary.csv")
    plot_embedding(Z, labels, results_dir / "fig_embedding.png", "WESAD latent wearable biosignal space")
    plot_similarity_boxplot(result, results_dir / "fig_similarity_boxplot.png", "WESAD stress/affect affinity")
    plot_permutation(result, results_dir / "fig_permutation_test.png", "WESAD permutation test")

    write_report(
        results_dir / "report.md",
        dataset_name="WESAD",
        hypothesis="Wearable-derived latent biosignal similarity is higher for subject-state samples sharing the same affective state than for samples from different states.",
        result=result,
        extra={
            "usable_subject_state_samples": int(len(rows)),
            "states": sorted(pd.unique(labels).tolist()),
            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        },
    )

    print(f"\nDone. Results saved to: {results_dir.resolve()}")
