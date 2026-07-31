# Expected outputs

After a successful run, the following files are the main inputs for the revised manuscript:

```text
results_independence_controlled/
├── three_dataset_independence_controlled_summary.csv
├── three_dataset_summary.md
├── fig_three_dataset_independent_unit_delta.png
├── run_status.json
├── wesad/
│   ├── report.md
│   └── primary_cross_subject/
│       ├── summary.csv
│       ├── independent_unit_deltas.csv
│       ├── fig_subject_delta.png
│       └── fig_permutation.png
├── case/
│   ├── report.md
│   ├── target_summary_with_fdr.csv
│   ├── leave_one_video_out.csv
│   └── primary_valence_arousal_quadrant/
│       ├── summary.csv
│       └── independent_unit_deltas.csv
└── ptbxl/
    ├── report.md
    ├── primary_patient_independent/
    │   ├── summary.csv
    │   └── independent_unit_deltas.csv
    └── sensitivity_age_sex_adjusted/
        └── summary.csv
```

Do not copy the previous pair-level Δ or P values into the revised manuscript. The values in `three_dataset_independence_controlled_summary.csv` replace them.
