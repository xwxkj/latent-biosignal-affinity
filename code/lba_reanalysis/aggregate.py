from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .features import ensure_dir
from .reporting import plot_combined_unit_deltas


def aggregate_summaries(summaries: List[Dict[str, object]], results_dir: Path) -> pd.DataFrame:
    ensure_dir(results_dir)
    table = pd.DataFrame(summaries).sort_values("analysis_order")
    table.to_csv(results_dir / "three_dataset_independence_controlled_summary.csv", index=False)
    plot_combined_unit_deltas(table, results_dir / "fig_three_dataset_independent_unit_delta.png")
    cols = [
        "dataset", "n_independent_units", "n_samples", "mean_unit_delta",
        "bootstrap_ci_95_low", "bootstrap_ci_95_high", "permutation_p_two_sided",
        "permutation_p_greater", "hedges_g"
    ]
    summary_table = table[cols].copy()
    lines = ["# Three-dataset independence-controlled summary", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in summary_table.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("{:.6g}".format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (results_dir / "three_dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table
