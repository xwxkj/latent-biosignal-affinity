import numpy as np

from lba_reanalysis.stats import run_cluster_inference


def test_repeated_measure_effect_is_detected():
    rng = np.random.default_rng(1)
    n_subjects = 18
    states = np.tile(np.asarray(["baseline", "stress", "amusement"]), n_subjects)
    subjects = np.repeat(np.asarray(["S{:02d}".format(i) for i in range(n_subjects)]), 3)
    state_vectors = {
        "baseline": np.asarray([2.0, 0.0, 0.0, 0.0]),
        "stress": np.asarray([0.0, 2.0, 0.0, 0.0]),
        "amusement": np.asarray([0.0, 0.0, 2.0, 0.0]),
    }
    Z = np.vstack([state_vectors[state] + rng.normal(0, 0.25, 4) for state in states])
    result = run_cluster_inference(
        Z, states, subjects, metric="cosine", permutation_scheme="within_unit",
        n_permutations=499, n_bootstrap=499, seed=4
    )
    assert result["summary"]["mean_unit_delta"] > 0
    assert result["summary"]["permutation_p_greater"] < 0.05


def test_same_unit_and_same_block_are_excluded():
    rng = np.random.default_rng(2)
    subjects = np.repeat(["A", "B", "C", "D"], 3)
    blocks = np.tile(["v1", "v2", "v3"], 4)
    labels = np.tile(["low", "high", "low"], 4)
    Z = rng.normal(size=(12, 5))
    result = run_cluster_inference(
        Z, labels, subjects, blocks=blocks, metric="cosine",
        permutation_scheme="within_unit", n_permutations=99, n_bootstrap=99, seed=3
    )
    sample = result["sample_deltas"]
    assert (sample["same_reference_weight"] <= 5).all()
    assert result["summary"]["n_independent_units"] == 4
