# Manuscript Results Template

## Proof-of-concept analysis

To evaluate the latent biosignal affinity hypothesis, we conducted proof-of-concept analyses in pathological and affective settings using public biosignal datasets. For each dataset, raw signals were transformed into feature representations, standardized and projected into a low-dimensional latent biosignal space. We then computed pairwise cosine similarity between latent representations and tested whether same-affinity pairs showed higher similarity than different-affinity pairs.

## PTB-XL pathological affinity

PTB-XL was used to evaluate the pathological-affinity component. Each 12-lead ECG record was represented by statistical, dynamic and frequency-domain features extracted from the 100 Hz waveform. Diagnostic superclasses were used to define pathological affinity. Pairwise similarity in the latent ECG space was compared between pairs sharing the same diagnostic superclass and pairs from different diagnostic superclasses.

Manuscript-ready sentence after running:

> ECG-derived latent biosignal similarity was higher for pairs sharing the same diagnostic superclass than for pairs from different diagnostic groups (Δ = [INSERT], permutation p = [INSERT]), supporting the pathological-affinity component of the proposed framework.

## WESAD stress-affinity

WESAD was used to evaluate whether affective states form reproducible structures in wearable biosignal space. Chest-worn ECG, EDA, EMG, respiration, temperature and acceleration signals were summarized into subject-state representations for baseline, stress and amusement. Pairwise similarity was then compared within and across affective states.

Manuscript-ready sentence after running:

> Wearable-derived latent biosignal representations showed higher similarity within the same affective state than across different states (Δ = [INSERT], permutation p = [INSERT]), supporting the stress-affinity prediction of the framework.

## Interpretation standard

The result is considered supportive if:

1. Δ similarity > 0;
2. permutation p < 0.05;
3. the finding remains directionally stable under different latent dimensions and sample sizes;
4. confound-controlled or matched-pair analyses do not eliminate the effect.

## Caution

These analyses do not imply deterministic inference of individual personality, friendship or disease. They test whether group-level affinity labels are statistically associated with latent biosignal similarity.
