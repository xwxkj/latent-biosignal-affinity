from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import (
    FEATURE_VERSION,
    basic_signal_features,
    download_file,
    ensure_dir,
    residualize_matrix,
    robust_latent_embedding,
)
from .reporting import plot_permutation, plot_unit_deltas, save_result, write_markdown_report
from .stats import run_cluster_inference

PTBXL_BASE = "https://physionet.org/files/ptb-xl/1.0.3/"


def parse_header(hea_path: Path):
    lines = hea_path.read_text(errors="ignore").splitlines()
    first = lines[0].split()
    n_sig = int(first[1])
    fs = float(first[2])
    n_samples = int(first[3])
    sigs = []
    for line in lines[1:1+n_sig]:
        parts = line.split()
        gain = 1.0
        baseline = 0.0
        if len(parts) > 2:
            raw = parts[2].split("/")[0]
            if "(" in raw and ")" in raw:
                try:
                    gain = float(raw.split("(")[0])
                    baseline = float(raw.split("(")[1].split(")")[0])
                except Exception:
                    pass
            else:
                try:
                    gain = float(raw)
                except Exception:
                    pass
        sigs.append((gain, baseline))
    return n_sig, fs, n_samples, sigs


def read_wfdb_16(record_path_no_ext: Path) -> np.ndarray:
    hea_path = record_path_no_ext.with_suffix(".hea")
    dat_path = record_path_no_ext.with_suffix(".dat")
    n_sig, _, n_samples, sigs = parse_header(hea_path)
    raw = np.fromfile(dat_path, dtype="<i2")
    expected = n_samples * n_sig
    if raw.size < expected:
        raise ValueError("Unexpected data size in {}".format(dat_path))
    raw = raw[:expected].reshape(-1, n_sig)
    gains = np.asarray([item[0] if abs(item[0]) > 1e-12 else 1.0 for item in sigs], dtype=float)
    baselines = np.asarray([item[1] for item in sigs], dtype=float)
    return (raw.astype(float) - baselines) / gains


def _label_and_score(row: pd.Series, scp_map: pd.DataFrame) -> Tuple[Optional[str], float]:
    try:
        codes = ast.literal_eval(row["scp_codes"])
    except Exception:
        return None, float("nan")
    best_class = None
    best_score = -np.inf
    for code, score in codes.items():
        if code not in scp_map.index:
            continue
        statement = scp_map.loc[code]
        diagnostic = statement.get("diagnostic", False)
        if not (bool(diagnostic) is True or str(diagnostic).lower() == "true"):
            continue
        diagnostic_class = statement.get("diagnostic_class", None)
        if diagnostic_class is None or pd.isna(diagnostic_class) or str(diagnostic_class).strip() == "":
            continue
        try:
            confidence = float(score)
        except Exception:
            confidence = 0.0
        if confidence > best_score:
            best_class = str(diagnostic_class)
            best_score = confidence
    return best_class, float(best_score)


def _download_metadata(data_dir: Path) -> None:
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        download_file(PTBXL_BASE + name, data_dir / name)


def _download_records(data_dir: Path, filenames: List[str]) -> None:
    for rel in tqdm(filenames, desc="Ensuring PTB-XL records"):
        rel_no_ext = rel[:-4] if rel.endswith(".hea") else rel
        for ext in (".hea", ".dat"):
            path = data_dir / (rel_no_ext + ext)
            if path.exists() and path.stat().st_size > 0:
                continue
            ensure_dir(path.parent)
            download_file(PTBXL_BASE + rel_no_ext + ext, path, timeout=180)


def _select_one_record_per_patient(df: pd.DataFrame, max_records: int, seed: int) -> pd.DataFrame:
    required = ["patient_id", "filename_lr", "primary_label", "primary_score"]
    clean = df.dropna(subset=required).copy()
    clean["patient_id"] = clean["patient_id"].astype(str)
    clean = clean.reset_index().sort_values(
        ["patient_id", "primary_score", "ecg_id"], ascending=[True, False, True]
    )
    unique = clean.drop_duplicates(subset=["patient_id"], keep="first")
    n_classes = unique["primary_label"].nunique()
    per_class = max(1, int(max_records // n_classes))
    selected = []
    for label, group in unique.groupby("primary_label"):
        selected.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    return pd.concat(selected, ignore_index=True).sample(frac=1.0, random_state=seed).head(max_records)


def _age_sex_design(meta: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(meta.get("age", pd.Series(np.nan, index=meta.index)), errors="coerce").to_numpy(float)
    age_missing = ~np.isfinite(age)
    age_fill = np.nanmedian(age) if np.any(np.isfinite(age)) else 0.0
    age[age_missing] = age_fill
    age = (age - np.mean(age)) / max(np.std(age), 1e-12)
    sex = pd.to_numeric(meta.get("sex", pd.Series(np.nan, index=meta.index)), errors="coerce").to_numpy(float)
    sex_missing = ~np.isfinite(sex)
    sex[sex_missing] = np.nanmedian(sex) if np.any(np.isfinite(sex)) else 0.0
    return np.column_stack([age, sex, age_missing.astype(float), sex_missing.astype(float)])


def _age_sex_strata(meta: pd.DataFrame) -> np.ndarray:
    age = pd.to_numeric(meta.get("age", pd.Series(np.nan, index=meta.index)), errors="coerce")
    decade = (age // 10 * 10).fillna(-1).astype(int).astype(str)
    sex = pd.to_numeric(meta.get("sex", pd.Series(np.nan, index=meta.index)), errors="coerce").fillna(-1).astype(int).astype(str)
    return ("age" + decade + "_sex" + sex).to_numpy()


def _load_or_extract_features(
    data_dir: Path,
    selected: pd.DataFrame,
    max_records: int,
    download: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ensure_dir(data_dir / "cache_independence_controlled")
    feature_path = cache_dir / "ptbxl_features_{}_max{}.npy".format(FEATURE_VERSION, max_records)
    meta_path = cache_dir / "ptbxl_meta_{}_max{}.csv".format(FEATURE_VERSION, max_records)
    if feature_path.exists() and meta_path.exists():
        X = np.load(feature_path)
        meta = pd.read_csv(meta_path)
        required_columns = {"patient_id", "primary_label", "age", "sex"}
        if (
            len(meta) == X.shape[0]
            and meta["patient_id"].astype(str).nunique() == len(meta)
            and required_columns.issubset(set(meta.columns))
        ):
            print("Loaded PTB-XL independence-controlled feature cache.")
            return X, meta

    if download:
        _download_records(data_dir, selected["filename_lr"].tolist())

    features = []
    rows = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Extracting patient-independent PTB-XL features"):
        record = (data_dir / str(row["filename_lr"])).with_suffix("")
        try:
            waveform = read_wfdb_16(record)
            feature = np.concatenate([basic_signal_features(waveform[:, ch], fs=100.0) for ch in range(waveform.shape[1])])
        except Exception as exc:
            print("Warning: skipped ECG {}: {}".format(row["ecg_id"], exc))
            continue
        features.append(feature)
        rows.append(row.to_dict())
    if len(features) < 20:
        raise RuntimeError("Too few usable patient-independent PTB-XL records.")
    X = np.vstack(features)
    meta = pd.DataFrame(rows)
    np.save(feature_path, X)
    meta.to_csv(meta_path, index=False)
    return X, meta


def run_ptbxl_independent(
    data_dir: Path,
    results_dir: Path,
    download: bool = False,
    max_records: int = 8000,
    latent_dim: int = 8,
    n_permutations: int = 5000,
    n_bootstrap: int = 5000,
    sensitivity_permutations: int = 1000,
    sensitivity_bootstrap: int = 1000,
    seed: int = 20260730,
) -> Dict[str, object]:
    ensure_dir(data_dir)
    ensure_dir(results_dir)
    if download:
        _download_metadata(data_dir)
    database_path = data_dir / "ptbxl_database.csv"
    statements_path = data_dir / "scp_statements.csv"
    if not database_path.exists() or not statements_path.exists():
        raise FileNotFoundError("PTB-XL metadata is missing in {}".format(data_dir))

    database = pd.read_csv(database_path, index_col="ecg_id")
    statements = pd.read_csv(statements_path, index_col=0)
    label_score = database.apply(lambda row: _label_and_score(row, statements), axis=1)
    database["primary_label"] = [item[0] for item in label_score]
    database["primary_score"] = [item[1] for item in label_score]
    selected = _select_one_record_per_patient(database, max_records=max_records, seed=seed)
    X, meta = _load_or_extract_features(data_dir, selected, max_records=max_records, download=download)

    Z, _, pca, _ = robust_latent_embedding(X, latent_dim=latent_dim)
    labels = meta["primary_label"].astype(str).to_numpy()
    patients = meta["patient_id"].astype(str).to_numpy()
    primary_dir = ensure_dir(results_dir / "primary_patient_independent")
    primary = run_cluster_inference(
        Z=Z,
        labels=labels,
        units=patients,
        metric="cosine",
        permutation_scheme="global",
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    save_result(primary, primary_dir, "PTB-XL", "One ECG per patient")
    plot_unit_deltas(primary, primary_dir / "fig_patient_delta.png", "PTB-XL: patient-level affinity contrast")
    plot_permutation(primary, primary_dir / "fig_permutation.png", "PTB-XL: patient-label permutation")
    patient_label_meta = meta[["patient_id", "primary_label"]].copy()
    patient_label_meta["unit"] = patient_label_meta["patient_id"].astype(str)
    patient_label_meta = patient_label_meta.drop(columns=["patient_id"])
    primary["unit_deltas"].merge(patient_label_meta, on="unit", how="left").to_csv(
        primary_dir / "patient_deltas_with_labels.csv", index=False
    )
    pd.DataFrame(Z, columns=["z{}".format(i+1) for i in range(Z.shape[1])]).assign(
        patient_id=patients, primary_label=labels
    ).to_csv(primary_dir / "latent_embedding.csv", index=False)

    # Age- and sex-adjusted sensitivity.
    adjusted_dir = ensure_dir(results_dir / "sensitivity_age_sex_adjusted")
    Z_adjusted = residualize_matrix(Z, _age_sex_design(meta))
    adjusted = run_cluster_inference(
        Z=Z_adjusted,
        labels=labels,
        units=patients,
        metric="cosine",
        permutation_scheme="within_stratum",
        strata=_age_sex_strata(meta),
        n_permutations=sensitivity_permutations,
        n_bootstrap=sensitivity_bootstrap,
        seed=seed + 10,
    )
    save_result(adjusted, adjusted_dir, "PTB-XL", "Age/sex-adjusted sensitivity")
    plot_unit_deltas(adjusted, adjusted_dir / "fig_patient_delta.png", "PTB-XL: age/sex-adjusted patient contrast")

    sensitivity_rows = []
    for dim in (4, 8, 16):
        Z_dim, _, _, _ = robust_latent_embedding(X, latent_dim=dim)
        for metric in ("cosine", "correlation", "neg_sqeuclidean"):
            result = run_cluster_inference(
                Z=Z_dim,
                labels=labels,
                units=patients,
                metric=metric,
                permutation_scheme="global",
                n_permutations=sensitivity_permutations,
                n_bootstrap=sensitivity_bootstrap,
                seed=seed + dim * 100 + len(metric),
            )
            row = dict(result["summary"])
            row.update({"latent_dim": dim, "dataset": "PTB-XL"})
            sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "sensitivity_latent_dimension_and_metric.csv", index=False)

    extra = [
        "Only one ECG was retained per patient; same-patient comparisons were therefore impossible.",
        "An age/sex-adjusted, age/sex-stratified permutation sensitivity analysis was also performed.",
        "PCA dimensions 4, 8 and 16 and three similarity metrics were evaluated as robustness checks.",
    ]
    write_markdown_report(
        results_dir / "report.md",
        "PTB-XL",
        "Patient-independent clinical-state affinity",
        "Patients sharing the same diagnostic superclass have more similar ECG-derived latent representations than patients from different superclasses.",
        primary,
        extra_lines=extra,
    )
    summary = dict(primary["summary"])
    summary.update({
        "dataset": "PTB-XL",
        "analysis_order": 3,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "age_sex_adjusted_mean_delta": adjusted["summary"]["mean_unit_delta"],
        "age_sex_adjusted_p_two_sided": adjusted["summary"]["permutation_p_two_sided"],
    })
    return summary
