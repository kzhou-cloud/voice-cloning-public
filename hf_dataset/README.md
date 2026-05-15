---
language:
  - en
license: cc-by-nc-4.0
pretty_name: Voice "Cloning" is Style Transfer — Audio Dataset
tags:
  - audio
  - speech
  - voice-cloning
  - text-to-speech
  - non-native-english
  - perception
  - dataset
task_categories:
  - audio-classification
  - text-to-speech
size_categories:
  - 10K<n<100K
extra_gated_prompt: >-
  By accessing this dataset you confirm that you will use the data for
  non-commercial research only, and will not redistribute it under more
  permissive terms.
configs:
  - config_name: default
    data_files:
      - split: original
        path: original/**/*.wav
      - split: cloned
        path: cloned/**/*.wav
      - split: cloned_styles
        path: cloned_styles/**/*.wav
      - split: cloned_iterative
        path: cloned_iterative/**/*.wav
---

# Voice *"Cloning"* is Style Transfer — Audio Dataset

Companion dataset for the preprint
**"Voice *'Cloning'* is Style Transfer"** (Zhou, Bianchi, Bartelds, Pot, Kwon, Zou; 2026).
Code, notebooks, and reproduction figures live at
[github.com/kzhou-cloud/voice-cloning-public](https://github.com/kzhou-cloud/voice-cloning-public).

> 🎧 Listen to a small set of paired examples on the
> [project page](https://kzhou-cloud.github.io/voice-cloning-public/).

## What's in here

| Split | # files | Description |
|---|---:|---|
| `original` | 1,916 | Human recordings of the *Grandfather Passage* from 86 non-native English speakers, split into sentence-level clips |
| `cloned` | 2,270 | Step-0 voice clones generated from the sources using ChatterBox, Coqui-XTTS, and ElevenLabs V3 (cross-sentence cloning paradigm) |
| `cloned_styles` | 13,572 | ChatterBox / Coqui-XTTS / ElevenLabs clones generated under six different style settings (high/low similarity, expressiveness, temperature) — ablation for §4.1 of the paper |
| `cloned_iterative` | 37,926 | Clones of clones: 50 rounds of repeated ChatterBox cloning for 43 speakers × 9 sentences |
| **Total** | **55,684 wavs / 17 GB** | |

All speakers and annotators are referenced by anonymized IDs (`speaker_001..speaker_086`).
The mapping between anonymized IDs and the raw upstream Prolific IDs is **not** distributed.

## Directory layout

```text
original/preprocessed_sentences/sentence_{N}_valid/speaker_{NNN}_sentence_{N}.wav
cloned/{model}/cloned_{model}_speaker_{NNN}_sentence_{N}.wav
cloned_styles/tts_outputs_{style}/{model}/cloned_{model}_speaker_{NNN}_sentence_{N}.wav
cloned_iterative/tts_outputs_{R}/{model}/cloned_{model}_speaker_{NNN}_sentence_{N}_from{F}_round{R}.wav
```

- `{model}` ∈ `{chatterbox, coqui_xtts, elevenlabs}` (iterative only uses `chatterbox`)
- `{style}` ∈ `{high_similarity, low_similarity, high_expressiveness, low_expressiveness, high_temperature, low_temperature}`
- `{N}` is the sentence index in the Grandfather Passage, 1–9
- `{R}` is the iterative cloning round, 2–50 (round 1 lives in `cloned/`)
- `{F}` is the sentence index used as reference for cross-sentence cloning in the previous round

## Companion CSVs

Bundled alongside the audio under `metadata/`:

- `ratings.csv` — 5,070 paired Likert ratings (7 dimensions × ~720 paired clips × ~3 annotators each), collected from 249 monolingual US-English annotators via Prolific. Paper-side filtering reduces this to 4,000 ratings from 177 annotators.
- `speaker_demographics.csv` — age, sex, ethnicity, native language, self-reported foreign-accent strength for the 86 analyzed speakers.
- `speaker_accent_orig_vs_cloned.csv` — CommonAccent classifier predictions on source vs cloned audio per speaker.

All companion CSVs reference speakers/annotators by the same anonymized IDs.

## How to use

```python
from huggingface_hub import snapshot_download

# Download the whole dataset to a local directory:
path = snapshot_download(
    repo_id="kzhou/voice_cloning_style_transfer",
    repo_type="dataset",
)
print(path)
```

To use it with the analysis notebooks in the companion code repo:

```bash
git clone https://github.com/kzhou-cloud/voice-cloning-public
cd voice-cloning-public
# Point audio_paths.py at the downloaded snapshot:
export VOICE_CLONING_ORIGINAL_AUDIO=$path/original
export VOICE_CLONING_CLONED_AUDIO=$path/cloned
export VOICE_CLONING_CLONED_STYLE_AUDIO=$path/cloned_styles
export VOICE_CLONING_CLONED_ITERATIVE_AUDIO=$path/cloned_iterative
# Then run any of the notebooks:
jupyter notebook
```

The notebooks load CSVs from `data/` and audio paths via `audio_paths.py`.

## Provenance

- **Source recordings** were collected from 86 non-native English speakers recruited via Prolific, each reading the *Grandfather Passage* once. Audio was preprocessed (silence-trim, amplitude-normalize, sentence-level segmentation via Whisper-based forced alignment) before the cloning stage. See §3.1 of the paper.
- **Cloned audio** uses the *cross-sentence* paradigm: target sentence ℓ is generated using sentence ℓ−1 as reference (cyclic wrap), so the model must extract generalizable speaker features rather than copy phonetic content.
- **Iterative audio** repeats the same cross-sentence paradigm using the *previous round's* output as the reference. This generates a 50-round "clone of clone" trajectory for each speaker.

The full collection methodology, IRB-approved consent flow, and quality-control protocol are described in the paper.

## Ethics, license, and intended use

This dataset was approved by the Cornell University Institutional Review Board.
All participants gave informed consent for the use of their de-identified audio in
voice-cloning research and were paid $18/hour. Participants were informed that
their audio would be:

1. used to evaluate how natural and synthetic audio data differ,
2. presented to other online workers for perceptual annotation,
3. potentially used to train models that distinguish natural from synthetic speech,
4. shared anonymously online via a public research dataset under non-commercial terms.

### License: CC BY-NC 4.0

### Forbidden uses

This dataset is released for **non-commercial research only**. The following
uses are explicitly **not permitted**:

- Generating, enabling, or promoting hate speech, harassment, discrimination,
  misinformation, or culturally offensive content.
- **Beyond explicit research purposes, voice cloning, speaker impersonation, or
  the creation of synthetic voices intended to resemble or replicate any
  participant.**
- Attempting to identify, re-identify, or infer the identity of any participant,
  including attempts to extract personally identifiable information from the
  audio or associated metadata.
- Any commercial, for-profit, or revenue-generating use, including product
  development, advertising, or monetized services.
- Any use that misrepresents, stereotypes, or falsely attributes characteristics,
  language abilities, accents, or identities to the speakers.
- Redistribution of the dataset under terms that conflict with or weaken these
  restrictions.

### Privacy

If you are a study participant and want your audio removed from this dataset,
please contact the corresponding author and we will issue an updated revision
of the dataset omitting your recordings.

## Citation

```bibtex
@article{zhou2026voicecloning,
  title  = {Voice "Cloning" is Style Transfer},
  author = {Zhou, Kaitlyn and Bianchi, Federico and Bartelds, Martijn
            and Pot, Anna and Kwon, Yongchan and Zou, James},
  year   = {2026},
  note   = {Preprint}
}
```

## Contact

Kaitlyn Zhou — kaitlynz@cornell.edu
