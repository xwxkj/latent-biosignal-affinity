# Latent Biosignal Affinity — independence-controlled reanalysis (v0.2)

This update replaces the original pair-level proof-of-concept inference with participant- and patient-level analyses that explicitly respect repeated measures, shared stimuli and patient identity.

## Main changes

- **WESAD:** all within-participant pairs are excluded; state labels are permuted within participant; confidence intervals use participant-level cluster bootstrap resampling.
- **CASE:** same-participant and same-video pairs are excluded; labels are permuted within participant; video-centred and leave-one-video-out sensitivity analyses quantify stimulus dependence.
- **PTB-XL:** one ECG is retained per patient; inference is patient-level; an age- and sex-adjusted stratified sensitivity analysis is included.
- Pair counts are no longer treated as independent sample sizes.

## Primary results

| Dataset | Independent units | Mean Δ | Cluster-bootstrap 95% CI | Restricted-permutation P | Hedges g |
|---|---:|---:|---:|---:|---:|
| WESAD | 15 participants | 0.5636 | [0.4502, 0.6836] | 0.0002 | 4.707 |
| CASE | 30 participants | 0.0200 | [0.0031, 0.0382] | 0.0002 | 0.782 |
| PTB-XL | 7,491 patients | 0.1172 | [0.1109, 0.1238] | 0.0002 | 0.786 |

The CASE effect is stimulus-sensitive: after video-centering, mean Δ = −0.0043, 95% CI [−0.0176, 0.0111], P = 0.1698.

## Repository organization

- `code/`: independence-controlled pipeline, tests and figure scripts.
- `results_summary/`: aggregate outputs only. Raw human biosignals and patient/participant-level rows are not redistributed.
- `figures/`: revised overview, independent-unit distributions and robustness analyses.

## Data

Raw data must be obtained from the original PTB-XL, WESAD and CASE repositories. This release does not redistribute human-subject biosignals.

## Ethical scope

The analyses are group-level statistical tests of cross-person physiological similarity. They are not designed for deterministic individual-level inference of personality, compatibility or diagnosis.

## Reproducing manuscript Figures 2 and 3

Figures 2 and 3 are generated programmatically from the numerical outputs of
the independence-controlled pipeline. They are not generative-AI images. Use
`code/make_reanalysis_figures.py` after running `code/run_reanalysis.py`.
Detailed input-to-panel provenance and commands are provided in
`code/FIGURE_REPRODUCTION.md` and `results_summary/figure_source_manifest.csv`.

## Correction release v0.2.2

The CASE analysis was recomputed at the original 1,000-Hz sampling rate with no segment-specific temporal subsampling. Corrected aggregate results, preprocessing provenance and updated figures are provided in tag `v0.2.2-case-fullrate-correction`. See `README_v0.2.2.md` and `results_summary/CASE_FULLRATE_VALIDATION.md`.

