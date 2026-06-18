from __future__ import annotations

import ast
import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

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

PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"


def download_ptbxl_metadata(data_dir: Path) -> None:
    ensure_dir(data_dir)
    for name in ["ptbxl_database.csv", "scp_statements.csv"]:
        download_file(PTBXL_BASE + name, data_dir / name)


def parse_header(hea_path: Path):
    lines = hea_path.read_text(errors="ignore").splitlines()
    first = lines[0].split()
    record_name = first[0]
    n_sig = int(first[1])
    fs = float(first[2])
    n_samples = int(first[3])
    sigs = []
    for line in lines[1:1+n_sig]:
        parts = line.split()
        filename = parts[0]
        fmt = parts[1]
        gain = 1.0
        baseline = 0.0
        if len(parts) > 2:
            # Examples: 1000/mV, 1000.0(0)/mV, 200
            graw = parts[2]
            if "/" in graw:
                graw = graw.split("/")[0]
            if "(" in graw and ")" in graw:
                try:
                    gain = float(graw.split("(")[0])
                    baseline = float(graw.split("(")[1].split(")")[0])
                except Exception:
                    gain = 1.0
                    baseline = 0.0
            else:
                try:
                    gain = float(graw)
                except Exception:
                    gain = 1.0
        sigs.append((filename, fmt, gain, baseline))
    return record_name, n_sig, fs, n_samples, sigs


def read_wfdb_16(record_path_no_ext: Path) -> np.ndarray:
    """
    Minimal WFDB format-16 reader for PTB-XL low-resolution files.
    Avoids depending on wfdb package.
    """
    hea_path = record_path_no_ext.with_suffix(".hea")
    dat_path = record_path_no_ext.with_suffix(".dat")
    record_name, n_sig, fs, n_samples, sigs = parse_header(hea_path)
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)

    raw = np.fromfile(dat_path, dtype="<i2")
    expected = n_samples * n_sig
    if raw.size < expected:
        raise ValueError(f"Unexpected data size in {dat_path}: {raw.size} < {expected}")
    raw = raw[:expected].reshape(-1, n_sig)

    gains = np.array([s[2] if abs(s[2]) > 1e-12 else 1.0 for s in sigs], dtype=float)
    baselines = np.array([s[3] for s in sigs], dtype=float)
    x = (raw.astype(float) - baselines) / gains
    return x


def extract_ptbxl_label(row, scp_map: pd.DataFrame) -> str | None:
    try:
        codes = ast.literal_eval(row["scp_codes"])
    except Exception:
        return None
    best_class = None
    best_score = -1.0
    for code, score in codes.items():
        if code not in scp_map.index:
            continue
        diag = scp_map.loc[code]
        if bool(diag.get("diagnostic", False)) is not True and str(diag.get("diagnostic", "")).lower() != "true":
            continue
        cls = diag.get("diagnostic_class", None)
        if pd.isna(cls) or cls is None or str(cls).strip() == "":
            continue
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        if score_f > best_score:
            best_score = score_f
            best_class = str(cls)
    return best_class


def select_balanced_records(df: pd.DataFrame, max_records: int) -> pd.DataFrame:
    # Keep records with primary diagnostic label and filename_lr.
    df = df.dropna(subset=["primary_label", "filename_lr"]).copy()
    if df.empty:
        raise ValueError("No PTB-XL records with diagnostic labels.")
    per_class = max(1, max_records // max(1, df["primary_label"].nunique()))
    chunks = []
    rng = np.random.default_rng(20260617)
    for lab, g in df.groupby("primary_label"):
        g = g.sample(frac=1.0, random_state=20260617)
        chunks.append(g.head(per_class))
    out = pd.concat(chunks, axis=0).sample(frac=1.0, random_state=20260617)
    return out.head(max_records)


def download_ptbxl_record_files(data_dir: Path, filenames: List[str]) -> None:
    for rel in tqdm(filenames, desc="Downloading PTB-XL records"):
        rel_no_ext = rel[:-4] if rel.endswith(".hea") else rel
        for ext in [".hea", ".dat"]:
            url = PTBXL_BASE + rel_no_ext + ext
            path = data_dir / (rel_no_ext + ext)
            if path.exists() and path.stat().st_size > 0:
                continue
            ensure_dir(path.parent)
            try:
                download_file(url, path, timeout=120)
            except Exception as e:
                print(f"Warning: failed to download {url}: {e}")


def run_ptbxl(
    data_dir: Path,
    results_dir: Path,
    download: bool,
    max_records: int,
    latent_dim: int,
    n_permutations: int,
    max_pairs: int,
):
    ensure_dir(data_dir)
    ensure_dir(results_dir)

    if download:
        download_ptbxl_metadata(data_dir)

    meta_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"
    if not meta_path.exists() or not scp_path.exists():
        raise FileNotFoundError(
            "PTB-XL metadata missing. Run with --download or place ptbxl_database.csv and scp_statements.csv in data/ptbxl/."
        )

    df = pd.read_csv(meta_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    df["primary_label"] = df.apply(lambda r: extract_ptbxl_label(r, scp), axis=1)
    selected = select_balanced_records(df, max_records=max_records)

    if download:
        download_ptbxl_record_files(data_dir, selected["filename_lr"].tolist())

    feats = []
    labels = []
    ids = []
    missing = 0
    for ecg_id, row in tqdm(selected.iterrows(), total=len(selected), desc="Extracting PTB-XL features"):
        rel = row["filename_lr"]
        rec_path = data_dir / rel
        rec_no_ext = rec_path.with_suffix("")
        try:
            x = read_wfdb_16(rec_no_ext)
            # x shape: samples x leads
            f_all = []
            for ch in range(x.shape[1]):
                f_all.append(basic_signal_features(x[:, ch], fs=100.0))
            feats.append(np.concatenate(f_all))
            labels.append(row["primary_label"])
            ids.append(ecg_id)
        except Exception as e:
            missing += 1
            continue

    if len(feats) < 20:
        raise RuntimeError(
            f"Only {len(feats)} usable PTB-XL records found. Increase max_records or check downloads."
        )

    X = np.vstack(feats)
    labels = np.asarray(labels)
    ids = np.asarray(ids)

    Z, scaler, pca = make_latent(X, latent_dim=latent_dim)
    result = pairwise_affinity_test(
        Z, labels,
        max_pairs=max_pairs,
        n_permutations=n_permutations,
    )

    emb = pd.DataFrame(Z, columns=[f"z{k+1}" for k in range(Z.shape[1])])
    emb.insert(0, "ecg_id", ids)
    emb["primary_label"] = labels
    emb.to_csv(results_dir / "latent_embedding.csv", index=False)

    save_pairwise_summary(result, results_dir / "pairwise_summary.csv")
    plot_embedding(Z, labels, results_dir / "fig_embedding.png", "PTB-XL latent ECG biosignal space")
    plot_similarity_boxplot(result, results_dir / "fig_similarity_boxplot.png", "PTB-XL pathological affinity")
    plot_permutation(result, results_dir / "fig_permutation_test.png", "PTB-XL permutation test")

    write_report(
        results_dir / "report.md",
        dataset_name="PTB-XL",
        hypothesis="ECG-derived latent biosignal similarity is higher for records sharing the same diagnostic superclass than for records with different diagnostic superclasses.",
        result=result,
        extra={
            "selected_records": int(len(selected)),
            "usable_records": int(len(feats)),
            "missing_or_failed_records": int(missing),
            "labels": sorted(pd.unique(labels).tolist()),
            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        },
    )

    print(f"\nDone. Results saved to: {results_dir.resolve()}")
