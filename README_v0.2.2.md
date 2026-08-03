# v0.2.2 CASE full-rate correction

This update corrects the CASE preprocessing pipeline and associated numerical outputs.

## Correction

The previous implementation conditionally applied stride-based temporal subsampling to long CASE segments while retaining the original sampling-rate argument during feature calculation. The corrected pipeline processes every CASE physiological segment at the original common sampling rate of 1,000 Hz and applies no segment-specific temporal subsampling.

All CASE feature caches were deleted before recomputation. The validated latent-embedding SHA-256 changed from `713bd631da7142da925abfe5975c97f1c021786728d2e6372487fe3542d75cb0` to `2998cbbddf374d3aaef35117b47a4789fd4a8f192026fa76dbbe084a557f6b6c`, confirming that the full-rate features were recomputed rather than retrieved from a stale cache.

## Corrected CASE result

- Independent participants: 30
- Participant-video representations: 330
- Mean participant-level delta: 0.0212369
- Cluster-bootstrap 95% CI: [0.0024174, 0.0416002]
- Two-sided restricted-permutation P: 0.00019996
- Positive participant-level contrasts: 27/30
- Video-centred sensitivity: delta = -0.0032716, 95% CI [-0.0173282, 0.0135420], P = 0.176823
- Leave-one-video-out range: 0.0077971 to 0.0300567

The qualitative interpretation is unchanged: the primary cross-participant, cross-video CASE contrast is positive, but evidence for a positive contrast is not retained after video-centering.

## Included files

- corrected `code/lba_reanalysis/case.py`
- CASE preprocessing manifest and validation report
- corrected aggregate CASE and three-dataset summaries
- updated Figures 1, 2 and 3
