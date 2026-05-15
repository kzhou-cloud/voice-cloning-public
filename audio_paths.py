"""Where the audio files live.

Defaults point at ``audio_data/`` inside this repo — the exact layout
shipped with the dataset. Two pieces are configurable if you keep your
audio somewhere else:

  1.  ``*_ROOT``  — base directory for each of the 4 audio categories.
  2.  ``*_TEMPLATE`` — subpath/filename pattern inside that root.

Both can be overridden via env vars (``VOICE_CLONING_*_AUDIO`` and
``VOICE_CLONING_*_TEMPLATE``) or by editing the right-hand strings below.

After editing, run ``python3 audio_paths.py`` — it prints which roots
resolve to existing directories and shows a few example resolved paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ── 1. Audio roots ──────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_AUDIO = _HERE / "audio_data"

ORIGINAL_ROOT          = Path(os.environ.get("VOICE_CLONING_ORIGINAL_AUDIO",
                                              _AUDIO / "original"))
CLONED_ROOT            = Path(os.environ.get("VOICE_CLONING_CLONED_AUDIO",
                                              _AUDIO / "cloned"))
CLONED_STYLE_ROOT      = Path(os.environ.get("VOICE_CLONING_CLONED_STYLE_AUDIO",
                                              _AUDIO / "cloned_styles"))
CLONED_ITERATIVE_ROOT  = Path(os.environ.get("VOICE_CLONING_CLONED_ITERATIVE_AUDIO",
                                              _AUDIO / "cloned_iterative"))


# ── 2. Subpath templates ────────────────────────────────────────────────────
# Each template is a format-string evaluated relative to the matching root.
# Placeholders:
#   {speaker}        speaker id (anon or raw, see resolve_speaker())
#   {sentence}       sentence number, 1..9
#   {model}          chatterbox | coqui_xtts | elevenlabs
#   {style}          a style preset name (only for style audio)
#   {round}          iterative cloning round, 1..50
#   {from_sentence}  the sentence index used as reference (iterative only)

ORIGINAL_TEMPLATE         = os.environ.get("VOICE_CLONING_ORIGINAL_TEMPLATE",
                              "preprocessed_sentences/sentence_{sentence}_valid/{speaker}_sentence_{sentence}.wav")

CLONED_TEMPLATE           = os.environ.get("VOICE_CLONING_CLONED_TEMPLATE",
                              "{model}/cloned_{model}_{speaker}_sentence_{sentence}.wav")

CLONED_STYLE_TEMPLATE     = os.environ.get("VOICE_CLONING_CLONED_STYLE_TEMPLATE",
                              "tts_outputs_{style}/{model}/cloned_{model}_{speaker}_sentence_{sentence}.wav")

CLONED_ITERATIVE_TEMPLATE = os.environ.get("VOICE_CLONING_CLONED_ITERATIVE_TEMPLATE",
                              "tts_outputs_{round}/{model}/cloned_{model}_{speaker}_sentence_{sentence}_from{from_sentence}_round{round}.wav")


# ── 3. Speaker-ID translation (optional) ────────────────────────────────────
# Public CSVs use anonymous IDs (speaker_001 … speaker_086).
# If your audio filenames also use those anon IDs, leave this alone.
#
# If your audio filenames still use the raw upstream IDs, point this at a
# JSON map. Two formats are accepted:
#   { "speaker_001": "57334d4...", ... }                                (direct)
#   { "speakers": { "57334d4...": "speaker_001", ... }, ... }           (raw→anon)
# (the latter is what scripts in voice_cloning_root produce).

_SPEAKER_MAP_PATH: Optional[str] = os.environ.get("VOICE_CLONING_SPEAKER_MAP")

_cached_map: Optional[dict[str, str]] = None


def _load_speaker_map() -> dict[str, str]:
    global _cached_map
    if _cached_map is not None:
        return _cached_map
    if not _SPEAKER_MAP_PATH:
        _cached_map = {}
        return _cached_map
    p = Path(_SPEAKER_MAP_PATH).expanduser()
    if not p.is_file():
        _cached_map = {}
        return _cached_map
    raw = json.loads(p.read_text())
    if "speakers" in raw and isinstance(raw["speakers"], dict):
        _cached_map = {anon: real for real, anon in raw["speakers"].items()}
    else:
        _cached_map = dict(raw)
    return _cached_map


def resolve_speaker(speaker_id: str) -> str:
    """Translate an anonymous speaker_id (speaker_001..) to whatever string
    the audio filenames actually use. If no mapping is configured, returns
    the input unchanged."""
    return _load_speaker_map().get(speaker_id, speaker_id)


# ── 4. Path builders ────────────────────────────────────────────────────────

def original_audio(speaker_id: str, sentence_num: int) -> Path:
    s = resolve_speaker(speaker_id)
    return ORIGINAL_ROOT / ORIGINAL_TEMPLATE.format(speaker=s, sentence=sentence_num)


def cloned_audio(model: str, speaker_id: str, sentence_num: int) -> Path:
    s = resolve_speaker(speaker_id)
    return CLONED_ROOT / CLONED_TEMPLATE.format(model=model, speaker=s, sentence=sentence_num)


def cloned_style_audio(model: str, style: str, speaker_id: str, sentence_num: int) -> Path:
    s = resolve_speaker(speaker_id)
    return CLONED_STYLE_ROOT / CLONED_STYLE_TEMPLATE.format(
        model=model, style=style, speaker=s, sentence=sentence_num)


def cloned_iterative_audio(model: str, round_idx: int, speaker_id: str,
                           sentence_num: int,
                           from_sentence: Optional[int] = None) -> Path:
    """Path to an iterative ('clone of clone') audio file at round R.

    ``round_idx`` is the iterative round, R ∈ {2,…,50}. (Round 1 lives in
    cloned/, not cloned_iterative/ — use ``cloned_audio`` for it.)

    ``from_sentence`` is the sentence index of the *previous round's* audio
    that was used as reference. Cross-sentence cloning wraps cyclically:
    each successive round shifts the reference by one, so for target
    sentence N at round R, the reference is sentence ``((N - R) % 9) + 1``
    of round R-1. If ``from_sentence`` is omitted, that formula is used.
    """
    if from_sentence is None:
        from_sentence = ((sentence_num - round_idx) % 9) + 1
    s = resolve_speaker(speaker_id)
    return CLONED_ITERATIVE_ROOT / CLONED_ITERATIVE_TEMPLATE.format(
        model=model, round=round_idx, speaker=s,
        sentence=sentence_num, from_sentence=from_sentence)


# ── 5. Print resolved roots (helpful for debugging) ─────────────────────────

def print_roots() -> None:
    rows = [
        ("ORIGINAL_ROOT",         ORIGINAL_ROOT),
        ("CLONED_ROOT",           CLONED_ROOT),
        ("CLONED_STYLE_ROOT",     CLONED_STYLE_ROOT),
        ("CLONED_ITERATIVE_ROOT", CLONED_ITERATIVE_ROOT),
    ]
    width = max(len(name) for name, _ in rows)
    print("Audio roots:")
    for name, path in rows:
        marker = "✓" if path.exists() else "✗ (missing)"
        print(f"  {name:<{width}}  =  {path}  {marker}")
    print()
    print("Templates:")
    for name, val in [
        ("ORIGINAL_TEMPLATE",         ORIGINAL_TEMPLATE),
        ("CLONED_TEMPLATE",           CLONED_TEMPLATE),
        ("CLONED_STYLE_TEMPLATE",     CLONED_STYLE_TEMPLATE),
        ("CLONED_ITERATIVE_TEMPLATE", CLONED_ITERATIVE_TEMPLATE),
    ]:
        print(f"  {name:<28}  =  {val}")
    print()
    if _SPEAKER_MAP_PATH:
        m = _load_speaker_map()
        print(f"Speaker map: {_SPEAKER_MAP_PATH}  ({len(m)} entries)")
    else:
        print("Speaker map: (not set — anon IDs used verbatim)")
    print()
    # Show a sample resolved path so the user knows it matches their layout
    print("Example resolved paths:")
    print(f"  original_audio('speaker_001', 3)")
    print(f"    → {original_audio('speaker_001', 3)}")
    print(f"  cloned_audio('chatterbox', 'speaker_001', 3)")
    print(f"    → {cloned_audio('chatterbox', 'speaker_001', 3)}")
    print(f"  cloned_iterative_audio('chatterbox', 25, 'speaker_001', 3, 2)")
    print(f"    → {cloned_iterative_audio('chatterbox', 25, 'speaker_001', 3, 2)}")


if __name__ == "__main__":
    print_roots()
