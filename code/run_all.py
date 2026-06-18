#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lba.ptbxl import run_ptbxl
from lba.wesad import run_wesad
from lba.amigos import run_amigos


def main():
    parser = argparse.ArgumentParser(
        description="Latent Biosignal Affinity proof-of-concept experiments."
    )
    parser.add_argument("--dataset", choices=["ptbxl", "wesad", "amigos"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--download", action="store_true", help="Download supported open datasets when possible.")
    parser.add_argument("--max-records", type=int, default=3000, help="PTB-XL maximum number of records.")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=200000, help="Maximum sampled pairs for pairwise tests.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ptbxl":
        run_ptbxl(
            data_dir=data_dir / "ptbxl",
            results_dir=results_dir / "ptbxl",
            download=args.download,
            max_records=args.max_records,
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            max_pairs=args.max_pairs,
        )
    elif args.dataset == "wesad":
        run_wesad(
            data_dir=data_dir / "wesad",
            zip_path=data_dir / "WESAD.zip",
            results_dir=results_dir / "wesad",
            download=args.download,
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            max_pairs=args.max_pairs,
        )
    elif args.dataset == "amigos":
        run_amigos(
            data_dir=data_dir / "amigos" / "data_preprocessed",
            results_dir=results_dir / "amigos",
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            max_pairs=args.max_pairs,
        )


if __name__ == "__main__":
    main()
