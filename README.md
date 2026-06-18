# Latent Biosignal Affinity

This repository contains code and aggregate proof-of-concept results for the manuscript:

**Latent biosignal affinity: a framework for studying shared human states through physiological similarity**

## Overview

The project evaluates whether samples sharing the same group-level affinity label show higher similarity in a low-dimensional latent biosignal space than samples with different labels.

Three public datasets are used:

1. **PTB-XL** — pathological affinity in 12-lead ECG.
2. **WESAD** — stress/affective affinity in wearable physiological signals.
3. **CASE** — psychological/affective affinity using valence-arousal annotations.

## Main proof-of-concept results

| Dataset | Affinity type | Delta similarity | Permutation p-value |
|---|---|---:|---:|
| PTB-XL | Pathological affinity | 0.1163 | 2.0e-4 |
| WESAD | Stress/affective affinity | 0.3303 | 2.0e-4 |
| CASE | Psychological/affective affinity | 0.0225 | 2.0e-4 |

## Repository contents

- code/: analysis scripts and reusable functions
- results_summary/: aggregate summaries and pairwise-test outputs
- figures/: proof-of-concept figures

## Data availability

This repository does not redistribute raw human-subject biosignal data. Raw datasets should be obtained from their original sources: PTB-XL from PhysioNet, WESAD from the UCI Machine Learning Repository / University of Siegen, and CASE from Scientific Data / Springer Nature Figshare.

## Code availability

The code reproduces the feature extraction, latent embedding, pairwise similarity analysis and permutation testing used for the proof-of-concept analyses.

## Ethical note

The analyses are intended for group-level statistical evaluation of latent biosignal affinity. They are not intended for deterministic individual-level inference of personality, relationship, health status, social compatibility or clinical diagnosis.
