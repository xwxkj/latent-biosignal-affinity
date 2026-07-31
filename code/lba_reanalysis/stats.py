from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260730


@dataclass
class EncodedDesign:
    labels: np.ndarray
    label_names: List[str]
    units: np.ndarray
    unit_names: List[str]
    blocks: Optional[np.ndarray]
    block_names: Optional[List[str]]
    strata: Optional[np.ndarray]
    stratum_names: Optional[List[str]]


def _encode(values: Sequence[object]) -> Tuple[np.ndarray, List[str]]:
    text = pd.Series(values).astype(str).to_numpy()
    names = sorted(pd.unique(text).tolist())
    mapping = {name: idx for idx, name in enumerate(names)}
    codes = np.asarray([mapping[value] for value in text], dtype=np.int64)
    return codes, names


def encode_design(
    labels: Sequence[object],
    units: Sequence[object],
    blocks: Optional[Sequence[object]] = None,
    strata: Optional[Sequence[object]] = None,
) -> EncodedDesign:
    label_codes, label_names = _encode(labels)
    unit_codes, unit_names = _encode(units)
    if blocks is None:
        block_codes, block_names = None, None
    else:
        block_codes, block_names = _encode(blocks)
    if strata is None:
        stratum_codes, stratum_names = None, None
    else:
        stratum_codes, stratum_names = _encode(strata)
    return EncodedDesign(
        labels=label_codes,
        label_names=label_names,
        units=unit_codes,
        unit_names=unit_names,
        blocks=block_codes,
        block_names=block_names,
        strata=stratum_codes,
        stratum_names=stratum_names,
    )


def _prepare_vectors(Z: np.ndarray, metric: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    if metric == "cosine":
        norms = np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z / norms
    if metric == "correlation":
        centred = Z - np.mean(Z, axis=1, keepdims=True)
        norms = np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
        return centred / norms
    if metric == "neg_sqeuclidean":
        return Z
    raise ValueError("Unsupported metric: {}".format(metric))


def _group_aggregates(
    X: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weights is None:
        weights = np.ones(X.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    counts = np.bincount(codes, weights=weights, minlength=n_groups).astype(np.float64)
    sums = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, X * weights[:, None])
    row_squares = np.sum(X * X, axis=1)
    sum_squares = np.bincount(codes, weights=row_squares * weights, minlength=n_groups).astype(np.float64)
    return counts, sums, sum_squares


def _similarity_to_group(
    X: np.ndarray,
    group_sum: np.ndarray,
    group_sum_squares: np.ndarray,
    group_count: np.ndarray,
    metric: str,
) -> np.ndarray:
    valid = group_count > 0
    result = np.full(X.shape[0], np.nan, dtype=np.float64)
    if metric in ("cosine", "correlation"):
        numer = np.einsum("ij,ij->i", X, group_sum)
        result[valid] = numer[valid] / group_count[valid]
    else:
        row_sq = np.sum(X * X, axis=1)
        mean_other_sq = np.zeros_like(group_count)
        mean_other_sq[valid] = group_sum_squares[valid] / group_count[valid]
        dot_mean = np.zeros_like(group_count)
        dot_mean[valid] = np.einsum("ij,ij->i", X[valid], group_sum[valid]) / group_count[valid]
        result[valid] = -(row_sq[valid] + mean_other_sq[valid] - 2.0 * dot_mean[valid])
    return result


def compute_unit_deltas(
    Z: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    blocks: Optional[np.ndarray] = None,
    metric: str = "cosine",
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compute exact sample- and unit-level affinity deltas while excluding pairs
    from the same independent unit and, optionally, the same stimulus/block.
    """
    X = _prepare_vectors(Z, metric)
    labels = np.asarray(labels, dtype=np.int64)
    units = np.asarray(units, dtype=np.int64)
    blocks_arr = None if blocks is None else np.asarray(blocks, dtype=np.int64)

    n, d = X.shape
    weights = np.ones(n, dtype=np.float64) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    if weights.shape[0] != n:
        raise ValueError("sample_weights must have one value per sample")
    n_labels = int(labels.max()) + 1
    n_units = int(units.max()) + 1

    all_sum = np.sum(X * weights[:, None], axis=0, keepdims=True)
    all_sum_sq = np.asarray([np.sum(np.sum(X * X, axis=1) * weights)], dtype=np.float64)
    all_count = np.asarray([float(np.sum(weights))], dtype=np.float64)

    unit_count, unit_sum, unit_sq = _group_aggregates(X, units, n_units, weights=weights)
    label_count, label_sum, label_sq = _group_aggregates(X, labels, n_labels, weights=weights)

    unit_label_codes = units * n_labels + labels
    n_unit_labels = n_units * n_labels
    ul_count, ul_sum, ul_sq = _group_aggregates(X, unit_label_codes, n_unit_labels, weights=weights)

    eligible_count = all_count[0] - unit_count[units]
    eligible_sum = all_sum[0] - unit_sum[units]
    eligible_sq = all_sum_sq[0] - unit_sq[units]

    same_count = label_count[labels] - ul_count[unit_label_codes]
    same_sum = label_sum[labels] - ul_sum[unit_label_codes]
    same_sq = label_sq[labels] - ul_sq[unit_label_codes]

    if blocks_arr is not None:
        n_blocks = int(blocks_arr.max()) + 1
        block_count, block_sum, block_sq = _group_aggregates(X, blocks_arr, n_blocks, weights=weights)
        unit_block_codes = units * n_blocks + blocks_arr
        n_unit_blocks = n_units * n_blocks
        ub_count, ub_sum, ub_sq = _group_aggregates(X, unit_block_codes, n_unit_blocks, weights=weights)

        block_label_codes = blocks_arr * n_labels + labels
        n_block_labels = n_blocks * n_labels
        bl_count, bl_sum, bl_sq = _group_aggregates(X, block_label_codes, n_block_labels, weights=weights)

        unit_block_label_codes = unit_block_codes * n_labels + labels
        n_ubl = n_unit_blocks * n_labels
        ubl_count, ubl_sum, ubl_sq = _group_aggregates(X, unit_block_label_codes, n_ubl, weights=weights)

        eligible_count = eligible_count - block_count[blocks_arr] + ub_count[unit_block_codes]
        eligible_sum = eligible_sum - block_sum[blocks_arr] + ub_sum[unit_block_codes]
        eligible_sq = eligible_sq - block_sq[blocks_arr] + ub_sq[unit_block_codes]

        same_count = same_count - bl_count[block_label_codes] + ubl_count[unit_block_label_codes]
        same_sum = same_sum - bl_sum[block_label_codes] + ubl_sum[unit_block_label_codes]
        same_sq = same_sq - bl_sq[block_label_codes] + ubl_sq[unit_block_label_codes]

    diff_count = eligible_count - same_count
    diff_sum = eligible_sum - same_sum
    diff_sq = eligible_sq - same_sq

    mean_same = _similarity_to_group(X, same_sum, same_sq, same_count, metric)
    mean_diff = _similarity_to_group(X, diff_sum, diff_sq, diff_count, metric)
    sample_delta = mean_same - mean_diff

    valid = np.isfinite(sample_delta) & (same_count > 0) & (diff_count > 0)
    sample_df = pd.DataFrame({
        "sample_index": np.arange(n, dtype=int),
        "unit_code": units,
        "label_code": labels,
        "mean_same": mean_same,
        "mean_different": mean_diff,
        "delta": sample_delta,
        "same_reference_weight": same_count,
        "different_reference_weight": diff_count,
        "valid": valid,
    })
    if blocks_arr is not None:
        sample_df["block_code"] = blocks_arr

    valid_units = units[valid]
    counts = np.bincount(valid_units, minlength=n_units).astype(np.float64)
    sum_delta = np.bincount(valid_units, weights=sample_delta[valid], minlength=n_units)
    sum_same = np.bincount(valid_units, weights=mean_same[valid], minlength=n_units)
    sum_diff = np.bincount(valid_units, weights=mean_diff[valid], minlength=n_units)
    present = counts > 0
    unit_df = pd.DataFrame({
        "unit_code": np.flatnonzero(present),
        "n_samples": counts[present].astype(int),
        "mean_same": sum_same[present] / counts[present],
        "mean_different": sum_diff[present] / counts[present],
        "delta": sum_delta[present] / counts[present],
    })

    pair_weight_same = np.sum(same_count[valid])
    pair_weight_diff = np.sum(diff_count[valid])
    descriptive = {
        "sample_weighted_mean_same": float(np.sum(mean_same[valid] * same_count[valid]) / max(pair_weight_same, 1.0)),
        "sample_weighted_mean_different": float(np.sum(mean_diff[valid] * diff_count[valid]) / max(pair_weight_diff, 1.0)),
        "n_valid_samples": int(np.sum(valid)),
        "n_independent_units": int(unit_df.shape[0]),
        "directed_same_comparisons": float(pair_weight_same),
        "directed_different_comparisons": float(pair_weight_diff),
    }
    return sample_df, unit_df, descriptive


def _index_groups(codes: np.ndarray) -> List[np.ndarray]:
    return [np.flatnonzero(codes == code) for code in np.unique(codes)]


def permute_labels(
    labels: np.ndarray,
    rng: np.random.Generator,
    scheme: str,
    units: np.ndarray,
    strata: Optional[np.ndarray] = None,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    out = labels.copy()
    if scheme == "global":
        return rng.permutation(labels)
    if scheme == "within_unit":
        for idx in _index_groups(units):
            out[idx] = rng.permutation(labels[idx])
        return out
    if scheme == "within_stratum":
        if strata is None:
            raise ValueError("strata are required for within_stratum permutation")
        for idx in _index_groups(strata):
            if idx.size > 1:
                out[idx] = rng.permutation(labels[idx])
        return out
    raise ValueError("Unknown permutation scheme: {}".format(scheme))


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, np.ndarray]:
    """Simple bootstrap retained for small auxiliary summaries."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan"), np.full(n_bootstrap, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draw = rng.choice(values, size=values.size, replace=True)
        boot[i] = float(np.mean(draw))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), boot


def cluster_bootstrap_distribution(
    Z: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    blocks: Optional[np.ndarray],
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """
    Multinomial cluster bootstrap. Independent units receive bootstrap
    multiplicities, reference-group means are recomputed with those weights,
    and comparisons within the same original unit remain excluded. This avoids
    treating duplicated bootstrap copies of the same participant as independent.
    """
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    units = np.asarray(units, dtype=np.int64)
    blocks_arr = None if blocks is None else np.asarray(blocks, dtype=np.int64)
    unique_units = np.unique(units)
    n_units = len(unique_units)
    if not np.array_equal(unique_units, np.arange(n_units)):
        remap = {int(code): idx for idx, code in enumerate(unique_units)}
        units_for_weights = np.asarray([remap[int(code)] for code in units], dtype=np.int64)
    else:
        units_for_weights = units
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_bootstrap, dtype=np.float64)
    probabilities = np.full(n_units, 1.0 / n_units)
    for iteration in range(n_bootstrap):
        unit_weights = rng.multinomial(n_units, probabilities).astype(np.float64)
        sample_weights = unit_weights[units_for_weights]
        _, unit_df, _ = compute_unit_deltas(
            Z=Z,
            labels=labels,
            units=units,
            blocks=blocks_arr,
            metric=metric,
            sample_weights=sample_weights,
        )
        codes = unit_df["unit_code"].to_numpy(dtype=int)
        weights_present = unit_weights[codes]
        valid = (weights_present > 0) & np.isfinite(unit_df["delta"].to_numpy(dtype=float))
        if not np.any(valid):
            distribution[iteration] = np.nan
        else:
            distribution[iteration] = float(np.average(
                unit_df.loc[valid, "delta"].to_numpy(dtype=float),
                weights=weights_present[valid],
            ))
    return distribution


def run_cluster_inference(
    Z: np.ndarray,
    labels: Sequence[object],
    units: Sequence[object],
    blocks: Optional[Sequence[object]] = None,
    strata: Optional[Sequence[object]] = None,
    metric: str = "cosine",
    permutation_scheme: str = "global",
    n_permutations: int = 5000,
    n_bootstrap: int = 5000,
    seed: int = SEED,
) -> Dict[str, object]:
    design = encode_design(labels=labels, units=units, blocks=blocks, strata=strata)
    sample_df, unit_df, descriptive = compute_unit_deltas(
        Z=Z,
        labels=design.labels,
        units=design.units,
        blocks=design.blocks,
        metric=metric,
    )
    observed = float(unit_df["delta"].mean())
    unit_values = unit_df["delta"].to_numpy(dtype=np.float64)
    bootstrap_dist = cluster_bootstrap_distribution(
        Z=np.asarray(Z, dtype=np.float64),
        labels=design.labels,
        units=design.units,
        blocks=design.blocks,
        metric=metric,
        n_bootstrap=n_bootstrap,
        seed=seed + 101,
    )
    ci_low, ci_high = np.percentile(bootstrap_dist[np.isfinite(bootstrap_dist)], [2.5, 97.5])
    ci_low, ci_high = float(ci_low), float(ci_high)

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for iteration in range(n_permutations):
        permuted = permute_labels(
            labels=design.labels,
            rng=rng,
            scheme=permutation_scheme,
            units=design.units,
            strata=design.strata,
        )
        _, perm_unit_df, _ = compute_unit_deltas(
            Z=Z,
            labels=permuted,
            units=design.units,
            blocks=design.blocks,
            metric=metric,
        )
        null[iteration] = float(perm_unit_df["delta"].mean())

    p_greater = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    null_center = float(np.mean(null))
    p_two_sided = float((np.sum(np.abs(null - null_center) >= abs(observed - null_center)) + 1) / (n_permutations + 1))

    sd = float(np.std(unit_values, ddof=1)) if unit_values.size > 1 else float("nan")
    cohens_dz = float(observed / sd) if np.isfinite(sd) and sd > 0 else float("nan")
    n_units = int(unit_values.size)
    correction = 1.0 - 3.0 / max(4.0 * n_units - 5.0, 1.0)
    hedges_g = float(correction * cohens_dz) if np.isfinite(cohens_dz) else float("nan")
    try:
        wilcoxon_two = float(stats.wilcoxon(unit_values, alternative="two-sided", zero_method="wilcox").pvalue)
        wilcoxon_greater = float(stats.wilcoxon(unit_values, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        wilcoxon_two = float("nan")
        wilcoxon_greater = float("nan")

    unit_df = unit_df.copy()
    unit_df["unit"] = [design.unit_names[int(code)] for code in unit_df["unit_code"]]
    sample_df = sample_df.copy()
    sample_df["unit"] = [design.unit_names[int(code)] for code in sample_df["unit_code"]]
    sample_df["label"] = [design.label_names[int(code)] for code in sample_df["label_code"]]
    if design.blocks is not None and design.block_names is not None:
        sample_df["block"] = [design.block_names[int(code)] for code in sample_df["block_code"]]

    summary = {
        "metric": metric,
        "permutation_scheme": permutation_scheme,
        "n_independent_units": n_units,
        "n_samples": int(len(labels)),
        "mean_unit_delta": observed,
        "bootstrap_ci_95_low": ci_low,
        "bootstrap_ci_95_high": ci_high,
        "permutation_p_greater": p_greater,
        "permutation_p_two_sided": p_two_sided,
        "permutation_null_mean": null_center,
        "cohens_dz": cohens_dz,
        "hedges_g": hedges_g,
        "wilcoxon_p_two_sided": wilcoxon_two,
        "wilcoxon_p_greater": wilcoxon_greater,
    }
    summary.update(descriptive)
    return {
        "summary": summary,
        "sample_deltas": sample_df,
        "unit_deltas": unit_df,
        "permutation_null": null,
        "bootstrap_distribution": bootstrap_dist,
        "label_names": design.label_names,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(p))
    adjusted = np.empty_like(ranked)
    running = 1.0
    for idx in range(len(ranked) - 1, -1, -1):
        rank = idx + 1.0
        running = min(running, ranked[idx] * m / rank)
        adjusted[idx] = running
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out
