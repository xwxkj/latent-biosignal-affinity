#!/usr/bin/env python3
from pathlib import Path
from lba.case import run_case

if __name__ == "__main__":
    run_case(
        data_dir=Path("data/case"),
        zip_path=Path("data/CASE_dataset.zip"),
        results_dir=Path("results/case"),
        download=False,
        latent_dim=8,
        n_permutations=5000,
        max_pairs=200000,
    )
