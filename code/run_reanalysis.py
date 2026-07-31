#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from lba_reanalysis.aggregate import aggregate_summaries
from lba_reanalysis.case import run_case_independent
from lba_reanalysis.features import ensure_dir
from lba_reanalysis.ptbxl import run_ptbxl_independent
from lba_reanalysis.wesad import run_wesad_independent


def main() -> None:
    parser = argparse.ArgumentParser(description="Independence-controlled latent biosignal affinity reanalysis")
    parser.add_argument("--dataset", choices=["all", "wesad", "case", "ptbxl"], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results_independence_controlled")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--ptbxl-max-records", type=int, default=8000)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--sensitivity-permutations", type=int, default=1000)
    parser.add_argument("--sensitivity-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = ensure_dir(Path(args.results_dir))
    summaries = []
    errors = {}

    tasks = []
    if args.dataset in ("all", "wesad"):
        tasks.append(("WESAD", lambda: run_wesad_independent(
            data_dir=data_dir / "wesad",
            zip_path=data_dir / "WESAD.zip",
            results_dir=results_dir / "wesad",
            download=args.download_missing,
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            sensitivity_permutations=args.sensitivity_permutations,
            sensitivity_bootstrap=args.sensitivity_bootstrap,
            seed=args.seed,
        )))
    if args.dataset in ("all", "case"):
        tasks.append(("CASE", lambda: run_case_independent(
            data_dir=data_dir / "case",
            results_dir=results_dir / "case",
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            sensitivity_permutations=args.sensitivity_permutations,
            sensitivity_bootstrap=args.sensitivity_bootstrap,
            seed=args.seed,
        )))
    if args.dataset in ("all", "ptbxl"):
        tasks.append(("PTB-XL", lambda: run_ptbxl_independent(
            data_dir=data_dir / "ptbxl",
            results_dir=results_dir / "ptbxl",
            download=args.download_missing,
            max_records=args.ptbxl_max_records,
            latent_dim=args.latent_dim,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            sensitivity_permutations=args.sensitivity_permutations,
            sensitivity_bootstrap=args.sensitivity_bootstrap,
            seed=args.seed,
        )))

    for name, function in tasks:
        print("\n" + "=" * 78)
        print("Running {} independence-controlled analysis".format(name))
        print("=" * 78)
        try:
            summaries.append(function())
        except Exception as exc:
            errors[name] = {"error": str(exc), "traceback": traceback.format_exc()}
            print("ERROR in {}: {}".format(name, exc))

    if summaries:
        aggregate_summaries(summaries, results_dir)
    with open(results_dir / "run_status.json", "w", encoding="utf-8") as handle:
        json.dump({"completed": [item["dataset"] for item in summaries], "errors": errors}, handle, indent=2)
    if errors:
        print("\nOne or more analyses failed. See {}.".format(results_dir / "run_status.json"))
        raise SystemExit(1)
    print("\nAll requested analyses completed. Results: {}".format(results_dir.resolve()))


if __name__ == "__main__":
    main()
