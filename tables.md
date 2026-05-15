# Reproduced tables and numbers

Companion to figures in `figures/`. Numbers below are computed directly from `data/` and `cache/` — re-run `python make_tables.py` to regenerate.


## Cohort sizes

- Speakers analyzed: **86**

- Annotators after filter: **177**  _(paper: 177)_

- Distinct rated clips (speaker × sentence × model): **1461**

- Total ratings: **4,000**  _(paper: 4,000)_


## Table 1 — Speaker demographics

| Variable | Value |
|---|---|
| Sex — Male | 43 (50.0%) |
| Sex — Female | 43 (50.0%) |
| Age (years) | mean = 38.3,  SD = 10.7,  range = 19–64 |
| Self-reported accent | mean = 3.8,  SD = 3.2,  range = 0–10 |
| Unique language backgrounds | 22 |



## Per-dimension deltas — all base models (Fig 3)

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.063 | 2.445 | 0.382 | +18.5% |
| Intimate Convo | 2.167 | 2.544 | 0.377 | +17.4% |
| Customer Service | 2.155 | 2.585 | 0.429 | +19.9% |
| Warm | 2.439 | 2.787 | 0.348 | +14.3% |
| Trust | 2.488 | 2.947 | 0.459 | +18.4% |
| Native English | 2.725 | 3.623 | 0.899 | +33.0% |
| Humanlike | 3.031 | 3.466 | 0.435 | +14.3% |



### Per-dimension deltas — chatterbox alone (Fig 12)

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.013 | 2.417 | 0.405 | +20.1% |
| Intimate Convo | 2.087 | 2.36 | 0.272 | +13.1% |
| Customer Service | 2.115 | 2.558 | 0.442 | +20.9% |
| Warm | 2.455 | 2.74 | 0.285 | +11.6% |
| Trust | 2.482 | 2.922 | 0.44 | +17.7% |
| Native English | 2.915 | 3.603 | 0.688 | +23.6% |
| Humanlike | 2.958 | 3.328 | 0.37 | +12.5% |



### Per-dimension deltas — coqui_xtts alone (Fig 12)

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.047 | 2.317 | 0.27 | +13.2% |
| Intimate Convo | 2.108 | 2.396 | 0.289 | +13.7% |
| Customer Service | 2.004 | 2.249 | 0.245 | +12.2% |
| Warm | 2.336 | 2.564 | 0.228 | +9.8% |
| Trust | 2.413 | 2.706 | 0.292 | +12.1% |
| Native English | 2.626 | 3.566 | 0.94 | +35.8% |
| Humanlike | 3.064 | 3.415 | 0.351 | +11.5% |



### Per-dimension deltas — elevenlabs alone (Fig 12)

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.114 | 2.584 | 0.47 | +22.2% |
| Intimate Convo | 2.277 | 2.811 | 0.533 | +23.4% |
| Customer Service | 2.325 | 2.916 | 0.591 | +25.4% |
| Warm | 2.523 | 3.026 | 0.504 | +20.0% |
| Trust | 2.561 | 3.188 | 0.626 | +24.5% |
| Native English | 2.682 | 3.691 | 1.009 | +37.6% |
| Humanlike | 3.053 | 3.611 | 0.558 | +18.3% |



### Per-dimension deltas — ElevenLabs low-expressiveness (Fig 11)

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 1.876 | 2.456 | 0.58 | +30.9% |
| Intimate Convo | 2.228 | 2.834 | 0.606 | +27.2% |
| Customer Service | 2.048 | 2.668 | 0.62 | +30.3% |
| Warm | 2.35 | 2.816 | 0.466 | +19.8% |
| Trust | 2.434 | 3.098 | 0.664 | +27.3% |
| Native English | 2.738 | 3.852 | 1.114 | +40.7% |
| Humanlike | 3.09 | 3.74 | 0.65 | +21.0% |



## Per-dimension deltas, by speaker sex (Fig 13)


### Female

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.099 | 2.468 | 0.37 | +17.6% |
| Intimate Convo | 2.319 | 2.774 | 0.455 | +19.6% |
| Customer Service | 2.296 | 2.759 | 0.464 | +20.2% |
| Warm | 2.609 | 2.986 | 0.377 | +14.4% |
| Trust | 2.655 | 3.196 | 0.541 | +20.4% |
| Native English | 2.735 | 3.772 | 1.038 | +37.9% |
| Humanlike | 3.197 | 3.543 | 0.346 | +10.8% |



### Male

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.033 | 2.426 | 0.393 | +19.3% |
| Intimate Convo | 2.037 | 2.348 | 0.311 | +15.3% |
| Customer Service | 2.036 | 2.436 | 0.4 | +19.6% |
| Warm | 2.294 | 2.617 | 0.323 | +14.1% |
| Trust | 2.346 | 2.735 | 0.389 | +16.6% |
| Native English | 2.716 | 3.496 | 0.78 | +28.7% |
| Humanlike | 2.89 | 3.4 | 0.51 | +17.6% |



## Long vs short reference prompt — sentence 8 (Fig 9)

**Short-prompt (default 1-sentence reference):**

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.041 | 2.438 | 0.397 | +19.4% |
| Intimate Convo | 2.07 | 2.438 | 0.368 | +17.8% |
| Customer Service | 2.093 | 2.312 | 0.219 | +10.5% |
| Warm | 2.314 | 2.688 | 0.374 | +16.1% |
| Trust | 2.395 | 2.792 | 0.396 | +16.5% |
| Native English | 2.581 | 3.625 | 1.044 | +40.4% |
| Humanlike | 2.89 | 3.375 | 0.485 | +16.8% |


**Long-prompt (concatenated 7-sentence reference, ~37 s):**

| Dimension | Source | Cloned | Δ | % change |
|---|---|---|---|---|
| Authoritative | 2.041 | 2.347 | 0.306 | +15.0% |
| Intimate Convo | 2.07 | 2.516 | 0.446 | +21.6% |
| Customer Service | 2.093 | 2.718 | 0.625 | +29.8% |
| Warm | 2.314 | 2.815 | 0.501 | +21.6% |
| Trust | 2.395 | 2.847 | 0.451 | +18.8% |
| Native English | 2.581 | 3.395 | 0.814 | +31.5% |
| Humanlike | 2.89 | 3.379 | 0.489 | +16.9% |



## Accent reclassification after cloning (Fig 4)

| Model | N speakers | non-native → native (%) | native → non-native (%) |
|---|---|---|---|
| chatterbox | 86 | 16.3% | 7.0% |
| coqui_xtts | 86 | 20.9% | 0.0% |
| elevenlabs | 86 | 16.3% | 7.0% |



## Duration distribution + differential entropy (Fig 15)

| Group | Mean duration (s) | H (nats) | ΔH vs original |
|---|---|---|---|
| Original (pooled) | 5.045 | 2.06 | — |
| Cloned — chatterbox | 4.905 | 1.852 | -0.209 |
| Cloned — coqui_xtts | 5.624 | 1.902 | -0.158 |
| Cloned — elevenlabs | 5.091 | 1.944 | -0.117 |



## Speaker identity probe — Table 3

Random-forest, LightGBM and SVM speaker-classification accuracy on source vs cloned recordings. Train: 5 sentences per speaker; test: the remaining sentences. Run separately for source and cloned audio.

| Dataset | Classifier | Top-1 accuracy | Mean incorrect-spread (≥5%) | F→M | M→F |
|---|---|---|---|---|---|
| Source | Random Forest | 85% | 0.56 | 2.9% | 1.7% |
| Source | LightGBM | 68% | 1.16 | nan | nan |
| Source | SVM (RBF) | 80% | 0.72 | nan | nan |
| Cloned | Random Forest | 53% | 4.3 | 6.8% | 5.6% |
| Cloned | LightGBM | 50% | 4.86 | nan | nan |
| Cloned | SVM (RBF) | 55% | 3.79 | nan | nan |



## Iterative cloning summary (Chatterbox, 50 rounds, Fig 5 / 16)

- Speakers with all 9 sentences × 50 rounds present: **87**

- Bounding-sphere radius (centroid → farthest embedding):

  - Source (round 0): **409.8**  _(paper: 366)_

  - Round 50:         **340.1**  _(paper: 336)_

| Sex | CosineSim source ↔ round 50 | Mean F0 (Hz) — round 0 | Mean F0 (Hz) — round 50 |
|---|---|---|---|
| Female | 0.217 | 199.5 | 280.4 |
| Male | 0.296 | 128.9 | 259.3 |



**Emotion classifier probabilities, round 0 → 50 (NVIDIA Audio2Emotion-v3):**

| Sex | Emotion | Mean prob — round 0 | Mean prob — round 50 | Δ |
|---|---|---|---|---|
| Female | angry | 0.069 | 0.666 | 0.597 |
| Female | happy | 0.07 | 0.14 | 0.07 |
| Female | neutral | 0.535 | 0.046 | -0.49 |
| Female | sad | 0.224 | 0.016 | -0.208 |
| Male | angry | 0.043 | 0.562 | 0.519 |
| Male | happy | 0.016 | 0.245 | 0.228 |
| Male | neutral | 0.569 | 0.041 | -0.528 |
| Male | sad | 0.279 | 0.017 | -0.262 |



## Notes on the speaker probe (Table 3)

The probe is run on the **43 speakers with all 9 valid sentences** (paper
text claim — every speaker gets 5 train / 4 test sentences).

Note: the originally-published Table 3 numbers (85%/41%, 81%/38%, etc.)
were inadvertently computed on a larger cohort (n=83 — anyone with ≥6
sentences). The numbers above use the intended n=43 cohort, where every
speaker has exactly 5 train + 4 test sentences. These are what should
appear in Table 3 of the paper.

