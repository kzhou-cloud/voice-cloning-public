
#!/usr/bin/env python3
"""
Generate cloned speech from generation_plan.csv.

Reads the generation plan (built by preprocess_audio.ipynb) and runs TTS
voice-cloning jobs for each row whose input exists and hasn't been cloned yet.

Usage:
    python generate_speech.py --model chatterbox
    python generate_speech.py --model chatterbox --style high_style
    python generate_speech.py --model elevenlabs --style low_style
    python generate_speech.py --model all
    python generate_speech.py --list

Models: chatterbox, coqui_xtts, elevenlabs
Styles: base (default), high_style, low_style
"""

# Standard library
import argparse
import csv
import os
import subprocess
import sys
import warnings
import wave as wave_mod

_WORKER_MODE = os.environ.get("_GENERATE_SPEECH_WORKER") == "1"

if _WORKER_MODE:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import soundfile as sf
    import torch
    import torchaudio as ta

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        print("WARNING: No GPU detected. Running on CPU (will be slow).")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
DATA_PIPELINE = os.path.join(REPO_ROOT, "data_pipeline")
AUDIO_DIR = os.path.join(DATA_PIPELINE, "aidaform", "audio_files")
MODELS_DIR = os.path.join(ROOT, "open-sourced-tts")

ALL_MODELS = ["elevenlabs", "chatterbox", "coqui_xtts"]
ALL_STYLES = [
    "base",
    "high_expressiveness", "low_expressiveness",
    "high_similarity", "low_similarity",
    "high_temperature", "low_temperature",
]

# ---------------------------------------------------------------------------
# Style presets — parameters passed to each model's generate function.
#
#   Chatterbox (model.generate):
#     cfg_weight   (0.0–1.0, default 0.5): reference voice adherence.
#                  Higher = more faithful to reference, slower pacing.
#     exaggeration (0.0–2.0, default 0.5): emotional intensity.
#                  Higher = more dramatic delivery, tends to speed up.
#     temperature  (0.05–5.0, default 0.8): sampling randomness.
#                  Higher = more creative/variable; lower = more deterministic.
#     Docs: https://github.com/resemble-ai/chatterbox
#           https://www.mintlify.com/yocxy2/chatterboxyocxy/guides/configuration
#
#   ElevenLabs (via VoiceSettings):
#     stability        (0.0–1.0, default ~0.5): output consistency.
#                      Lower = more expressive/variable; higher = steadier.
#                      (closest analog to temperature — lower ≈ higher temp)
#     similarity_boost (0.0–1.0, default ~0.75): voice similarity to reference.
#                      Higher = closer match to original speaker.
#     style            (0.0–1.0, default 0.0): style exaggeration.
#                      Higher = more expressive delivery.
#     (no temperature parameter)
#     Docs: https://github.com/elevenlabs/skills/blob/main/text-to-speech/references/voice-settings.md
#           https://elevenlabs.io/docs/api-reference/voices/settings/update
#
#   Coqui XTTS (model.inference / tts_to_file **kwargs):
#     temperature (default 0.75): sampling randomness.
#                 Higher = more creative; lower = more deterministic.
#                 Very low (<0.3) can cause truncated or unnatural output.
#     Docs: https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "base": {
        "chatterbox": {},
        "elevenlabs": {},
        "coqui_xtts": {},
    },
    "high_expressiveness": {
        "chatterbox": {"exaggeration": 2},
        "elevenlabs": {"style": 1},
        "coqui_xtts": {},
    },
    "low_expressiveness": {
        "chatterbox": {"exaggeration": 0},
        "elevenlabs": {"style": 0},
        "coqui_xtts": {},
    },
    "high_similarity": {
        "chatterbox": {"cfg_weight": 1},
        "elevenlabs": {"similarity_boost": 1},
        "coqui_xtts": {},
    },
    "low_similarity": {
        "chatterbox": {"cfg_weight": 0.05},
        "elevenlabs": {"similarity_boost": 0},
        "coqui_xtts": {},
    },
    "high_temperature": {
        "chatterbox": {"temperature": 1},
        "elevenlabs": {"stability": 0},
        "coqui_xtts": {"temperature": 1},
    },
    "low_temperature": {
        "chatterbox": {"temperature": 0.05},
        "elevenlabs": {"stability": 1},
        "coqui_xtts": {"temperature": 0.45},
    },
}

VENV_MAP = {
    "chatterbox": "main",
    "coqui_xtts": "coqui",
    "elevenlabs": "main",
}

# These globals are set by main() based on --style
ACTIVE_STYLE = "base"
OUTPUT_DIR = ""
GENERATION_PLAN_PATH = ""


def _init_paths(style):
    """Set OUTPUT_DIR and GENERATION_PLAN_PATH based on style."""
    global ACTIVE_STYLE, OUTPUT_DIR, GENERATION_PLAN_PATH
    ACTIVE_STYLE = style
    if style == "base":
        OUTPUT_DIR = os.environ.get(
            "TTS_OUTPUT_DIR",
            os.path.join(ROOT, "tts_outputs"),
        )
        GENERATION_PLAN_PATH = os.path.join(ROOT, "generation_plan.csv")
    else:
        styled_dir = os.path.join(ROOT, "styled_voice_cloning")
        OUTPUT_DIR = os.environ.get(
            "TTS_OUTPUT_DIR",
            os.path.join(styled_dir, f"tts_outputs_{style}"),
        )
        GENERATION_PLAN_PATH = os.path.join(styled_dir, f"generation_plan_{style}.csv")


def _venv_python(venv_name):
    return os.path.join(ROOT, f".venv_{venv_name}", "bin", "python3")


def ref_stem(ref_path):
    """Extract a short identifier from the reference audio filename."""
    return os.path.splitext(os.path.basename(ref_path))[0]


def _read_plan():
    if not os.path.exists(GENERATION_PLAN_PATH):
        _create_plan_from_validated_audio()
    with open(GENERATION_PLAN_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


PREPROCESSED_DIR = os.path.join(AUDIO_DIR, "preprocessed_sentences")
SENTENCE_PAIRS = [(s, s % 9 + 1) for s in range(1, 10)]
FIELDNAMES = [
    "model_name", "participant_id",
    "input_clip", "input_text",
    "output_clip", "target_text",
    "predicted_output_clip",
    "missing_input_clip", "missing_output_clip", "cloned",
]

SENTENCE_TEXTS = {
    1: "You wish to know about my grandfather.",
    2: "Well, he is nearly 93 years old, yet he still thinks as swiftly as ever.",
    3: "He dresses himself in an old black frock coat, usually several buttons missing.",
    4: "A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect.",
    5: "When he speaks, his voice is just a bit cracked and quivers a bit.",
    6: "Twice each day he plays skillfully and with zest upon a small organ.",
    7: "Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day.",
    8: "We have often urged him to walk more and smoke less, but he always answers, \"Banana oil!\"",
    9: "Grandfather likes to be modern in his language.",
}


def _scan_validated_participants():
    """Scan preprocessed_sentences/ to discover all validated participants
    and which sentences each has available."""
    valid_map = {}
    for sent in range(1, 10):
        sent_dir = os.path.join(PREPROCESSED_DIR, f"sentence_{sent}_valid")
        if not os.path.isdir(sent_dir):
            continue
        for fname in os.listdir(sent_dir):
            if not fname.endswith(".wav"):
                continue
            pid = fname.replace(f"_sentence_{sent}.wav", "")
            valid_map.setdefault(pid, set()).add(sent)

    long_dir = os.path.join(PREPROCESSED_DIR, "sentence_long")
    long_pids = set()
    if os.path.isdir(long_dir):
        for fname in os.listdir(long_dir):
            if fname.endswith(".wav"):
                pid = fname.replace("_sentence_long.wav", "")
                long_pids.add(pid)

    return valid_map, long_pids


def _build_plan_rows(valid_map, long_pids, output_dir_name):
    """Build generation plan rows for all models × participants × sentence pairs."""
    rows = []
    for model in ALL_MODELS:
        for pid in sorted(valid_map):
            valid = valid_map[pid]
            for src_sent, tgt_sent in SENTENCE_PAIRS:
                ref_rel = f"aidaform/audio_files/preprocessed_sentences/sentence_{src_sent}_valid/{pid}_sentence_{src_sent}.wav"
                out_rel = f"aidaform/audio_files/preprocessed_sentences/sentence_{tgt_sent}_valid/{pid}_sentence_{tgt_sent}.wav"
                pred_rel = f"{output_dir_name}/{model}/cloned_{model}_{pid}_sentence_{tgt_sent}.wav"

                pred_abs = os.path.join(ROOT, pred_rel)

                rows.append({
                    "model_name": model,
                    "participant_id": pid,
                    "input_clip": ref_rel,
                    "input_text": SENTENCE_TEXTS.get(src_sent, ""),
                    "output_clip": out_rel,
                    "target_text": SENTENCE_TEXTS.get(tgt_sent, ""),
                    "predicted_output_clip": pred_rel,
                    "missing_input_clip": str(int(src_sent not in valid)),
                    "missing_output_clip": str(int(tgt_sent not in valid)),
                    "cloned": str(int(os.path.exists(pred_abs))),
                })

            has_long = pid in long_pids
            if has_long or all(s in valid for s in range(1, 8)):
                long_ref = f"aidaform/audio_files/preprocessed_sentences/sentence_long/{pid}_sentence_long.wav"
                out_rel = f"aidaform/audio_files/preprocessed_sentences/sentence_8_valid/{pid}_sentence_8.wav"
                pred_rel = f"{output_dir_name}/{model}/cloned_{model}_{pid}_sentence_8_long.wav"
                pred_abs = os.path.join(ROOT, pred_rel)

                rows.append({
                    "model_name": model,
                    "participant_id": pid,
                    "input_clip": long_ref,
                    "input_text": " ".join(SENTENCE_TEXTS[s] for s in range(1, 8)),
                    "output_clip": out_rel,
                    "target_text": SENTENCE_TEXTS.get(8, ""),
                    "predicted_output_clip": pred_rel,
                    "missing_input_clip": str(int(not has_long)),
                    "missing_output_clip": str(int(8 not in valid)),
                    "cloned": str(int(os.path.exists(pred_abs))),
                })
    return rows


def _create_plan_from_validated_audio():
    """Build a generation plan by scanning preprocessed_sentences/ for all
    validated participants, then write the style-specific plan CSV.

    Also refreshes the canonical source plan (base tts_outputs/ paths) so
    it stays in sync with what's actually on disk.
    """
    valid_map, long_pids = _scan_validated_participants()
    if not valid_map:
        print("ERROR: No validated audio found in "
              f"{PREPROCESSED_DIR}", file=sys.stderr)
        print("  Run the preprocess_audio notebook first.", file=sys.stderr)
        sys.exit(1)

    n_participants = len(valid_map)

    suffix = f"_{ACTIVE_STYLE}" if ACTIVE_STYLE != "base" else ""
    output_dir_name = f"tts_outputs{suffix}"
    rows = _build_plan_rows(valid_map, long_pids, output_dir_name)

    with open(GENERATION_PLAN_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {GENERATION_PLAN_PATH} from {n_participants} validated "
          f"participants ({len(rows)} rows)")


def _write_plan(fieldnames, rows):
    with open(GENERATION_PLAN_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_csv_path(rel_path):
    """Resolve a relative path from generation_plan.csv.

    Paths starting with 'aidaform/' refer to data_pipeline/aidaform/;
    all others (e.g. tts_outputs/) are relative to ROOT (voice_cloning/).
    """
    if rel_path.startswith("aidaform/"):
        return os.path.join(DATA_PIPELINE, rel_path)
    return os.path.join(ROOT, rel_path)


def sync_plan():
    """Update the cloned column in generation_plan.csv based on file existence."""
    fieldnames, rows = _read_plan()
    changed = 0
    for row in rows:
        exists = os.path.exists(_resolve_csv_path(row["predicted_output_clip"]))
        new_val = "1" if exists else "0"
        if row["cloned"] != new_val:
            row["cloned"] = new_val
            changed += 1
    _write_plan(fieldnames, rows)

    cloned = sum(1 for r in rows if r["cloned"] == "1")
    print(f"Plan synced: {cloned}/{len(rows)} cloned ({changed} updated)")


def load_jobs(model_name):
    """Load generation plan and return pending jobs for the given model.

    Skips rows where the input clip is missing or the output already exists.
    Returns list of (ref_audio, target_text, output_path) tuples.
    """
    _, rows = _read_plan()

    total = sum(1 for r in rows if r["model_name"] == model_name)
    jobs = []
    skipped_missing = 0
    skipped_done = 0

    for row in rows:
        if row["model_name"] != model_name:
            continue
        if row["missing_input_clip"] == "1":
            skipped_missing += 1
            continue

        ref_audio = _resolve_csv_path(row["input_clip"])
        out_path = _resolve_csv_path(row["predicted_output_clip"])

        if os.path.exists(out_path):
            skipped_done += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        jobs.append((ref_audio, row["target_text"], out_path))

    print(f"Plan for {model_name}: {len(jobs)} to generate, "
          f"{skipped_done} already done, {skipped_missing} missing input "
          f"(of {total} total)")
    return jobs


# ---------------------------------------------------------------------------
# Model runners — each takes a list of (ref_audio, text, out_path) jobs
# ---------------------------------------------------------------------------

def run_chatterbox(jobs):
    from chatterbox.tts import ChatterboxTTS

    params = STYLE_PRESETS[ACTIVE_STYLE]["chatterbox"]
    print(f"  Loading ChatterboxTTS model...  style={ACTIVE_STYLE}  params={params}")
    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    for i, (ref_audio, text, out_path) in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] ref={os.path.basename(ref_audio)}")
        wav = model.generate(text, audio_prompt_path=ref_audio, **params)
        ta.save(out_path, wav, model.sr)
        print(f"    ✓ {out_path}")


def run_coqui_xtts(jobs):
    import numpy as np
    from TTS.api import TTS as CoquiTTS

    params = STYLE_PRESETS[ACTIVE_STYLE].get("coqui_xtts", {})
    if ACTIVE_STYLE != "base" and not params:
        print(f"  NOTE: Coqui XTTS has no parameters for --style {ACTIVE_STYLE}; "
              f"output will be identical to base.")
    elif params:
        print(f"  style={ACTIVE_STYLE}  params={params}")

    os.environ["COQUI_TOS_AGREED"] = "1"

    _original_torch_load = torch.load
    _original_torchaudio_load = ta.load

    def _patched_torchaudio_load(filepath, *args, **kwargs):
        data, sr = sf.read(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        return torch.from_numpy(data.astype(np.float32)), sr

    torch.load = lambda *args, **kwargs: _original_torch_load(
        *args, **{**kwargs, "weights_only": False}
    )
    ta.load = _patched_torchaudio_load

    print("  Loading XTTS-v2 model...")
    tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(DEVICE)

    torch.load = _original_torch_load

    for i, (ref_audio, text, out_path) in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] ref={os.path.basename(ref_audio)}")
        try:
            tts.tts_to_file(
                text=text,
                file_path=out_path,
                speaker_wav=ref_audio,
                language="en",
                **params,
            )
            print(f"    ✓ {out_path}")
        except Exception as e:
            print(f"    SKIP (error: {e})")

    ta.load = _original_torchaudio_load


def _eleven_setup():
    """Shared ElevenLabs client setup. Returns (client, retry, print_quota,
    is_voice_limit_error) or None on missing API key."""
    import time
    from elevenlabs.client import ElevenLabs

    MAX_RETRIES = 5
    BASE_DELAY = 1.0

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        key_file = os.path.join(ROOT, "eleven_labs_api.txt")
        if os.path.exists(key_file):
            with open(key_file) as f:
                api_key = f.read().strip()
    if not api_key:
        print("  ERROR: Set ELEVENLABS_API_KEY or create eleven_labs_api.txt")
        return None

    client = ElevenLabs(api_key=api_key)

    def print_quota():
        try:
            sub = client.user.get().subscription
            print(f"  Quota: voices={sub.voice_slots_used}/{sub.voice_limit}  "
                  f"ops={sub.voice_add_edit_counter}/{sub.max_voice_add_edits}  "
                  f"chars={sub.character_count}/{sub.character_limit}")
        except Exception:
            pass

    def retry(fn, label):
        for attempt in range(MAX_RETRIES):
            try:
                return fn()
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429:
                    wait = BASE_DELAY * (2 ** attempt)
                    print(f"    Rate limited ({label}), waiting {wait:.0f}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Rate limit exceeded after {MAX_RETRIES} retries")

    def is_voice_limit_error(exc):
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, dict) and detail.get("status") == "voice_limit_reached":
                return True
        return "voice_limit_reached" in str(exc)

    return client, retry, print_quota, is_voice_limit_error


def run_elevenlabs(jobs):
    from io import BytesIO

    params = STYLE_PRESETS[ACTIVE_STYLE]["elevenlabs"]

    setup = _eleven_setup()
    if setup is None:
        return
    client, _api_call_with_retry, print_quota, _is_voice_limit_error = setup

    voice_settings = None
    if params:
        from elevenlabs import VoiceSettings
        voice_settings = VoiceSettings(**params)

    print(f"  style={ACTIVE_STYLE}  params={params}")
    print_quota()

    existing_voices = {}
    try:
        for v in client.voices.get_all().voices:
            existing_voices[v.name] = v.voice_id
        print(f"  Found {len(existing_voices)} existing voices in library")
    except Exception as e:
        print(f"  WARNING: Could not fetch voice library ({e}), will create all voices")

    # Voices we still want to reuse this run (matches a pending job's ref).
    needed_names = {f"clone_{ref_stem(j[0])}" for j in jobs}
    # Only clone_* voices are user-owned and deletable; marketplace voices
    # in get_all() must be excluded from eviction candidates.
    eviction_pool = [
        n for n in existing_voices
        if n.startswith("clone_") and n not in needed_names
    ]

    def _evict_one_voice(reason):
        """Delete one user-owned clone_* voice to free a slot."""
        nonlocal eviction_pool
        while eviction_pool:
            name = eviction_pool.pop(0)
            vid = existing_voices.pop(name, None)
            if vid is None:
                continue
            try:
                client.voices.delete(voice_id=vid)
                print(f"    Evicted voice '{name}' to free slot ({reason})")
                return True
            except Exception as e:
                print(f"    WARNING: failed to delete '{name}': {e}")
        fallback = [n for n in existing_voices if n.startswith("clone_")]
        if fallback:
            name = fallback[0]
            vid = existing_voices.pop(name)
            try:
                client.voices.delete(voice_id=vid)
                print(f"    Evicted reusable voice '{name}' to free slot ({reason})")
                return True
            except Exception as e:
                print(f"    WARNING: failed to delete '{name}': {e}")
        try:
            for v in client.voices.get_all().voices:
                existing_voices.setdefault(v.name, v.voice_id)
            fallback = [n for n in existing_voices if n.startswith("clone_")]
            if fallback:
                name = fallback[0]
                vid = existing_voices.pop(name)
                client.voices.delete(voice_id=vid)
                print(f"    Evicted voice '{name}' after refresh ({reason})")
                return True
        except Exception as e:
            print(f"    WARNING: refresh+delete failed: {e}")
        return False

    succeeded = 0
    skipped = 0
    deleted_after = 0
    for i, (ref_audio, text, out_path) in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] ref={os.path.basename(ref_audio)}")
        created_voice_id = None
        try:
            voice_name = f"clone_{ref_stem(ref_audio)}"
            if voice_name in existing_voices:
                voice_id = existing_voices[voice_name]
                print(f"    Reusing existing voice '{voice_name}'")
            else:
                def _create_voice(ra=ref_audio, vn=voice_name):
                    with open(ra, "rb") as f:
                        return client.voices.ivc.create(
                            name=vn, files=[BytesIO(f.read())])
                try:
                    voice = _api_call_with_retry(_create_voice, "create voice")
                except Exception as e:
                    if not _is_voice_limit_error(e):
                        raise
                    print(f"    voice_limit_reached on create, evicting one voice...")
                    if not _evict_one_voice("voice_limit_reached"):
                        raise RuntimeError(
                            "voice_limit_reached and no voices available to evict")
                    voice = _api_call_with_retry(
                        _create_voice, "create voice (after evict)")
                voice_id = voice.voice_id
                created_voice_id = voice_id
                existing_voices[voice_name] = voice_id
                print(f"    Created new voice '{voice_name}'")

            convert_kwargs = dict(
                voice_id=voice_id,
                text=text,
                model_id="eleven_v3",
                output_format="pcm_24000",
            )
            if voice_settings is not None:
                convert_kwargs["voice_settings"] = voice_settings

            def _tts(kw=convert_kwargs):
                return b"".join(client.text_to_speech.convert(**kw))
            pcm_data = _api_call_with_retry(_tts, "tts")

            with wave_mod.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm_data)
            succeeded += 1
            print(f"    ✓ {out_path}")
        except Exception as e:
            skipped += 1
            print(f"    SKIP (error: {e})")
        finally:
            # Roll: if we created the voice for this job, delete it now so the
            # slot is free for the next job. Voices we reused are left alone.
            if created_voice_id is not None:
                try:
                    client.voices.delete(voice_id=created_voice_id)
                    existing_voices.pop(voice_name, None)
                    deleted_after += 1
                except Exception as e:
                    print(f"    WARNING: post-synth delete of '{voice_name}' "
                          f"failed: {e}")
        if i % 50 == 0:
            print_quota()
            print(f"  Progress: {succeeded} succeeded, {skipped} skipped, "
                  f"{deleted_after} deleted out of {i}")

    print(f"\n  Final: {succeeded} succeeded, {skipped} skipped, "
          f"{deleted_after} voices deleted out of {len(jobs)}")
    print_quota()


def run_elevenlabs_all_styles(jobs_by_style):
    """Generate ElevenLabs outputs for ALL styles per ref in one voice lifecycle.

    jobs_by_style: dict[style_name -> list of (ref_audio, target_text, out_path)]

    Strategy: group every job (across all styles) by ref_audio. For each ref:
        1. Reuse the matching `clone_<ref_stem>` voice if it already exists,
           otherwise create it (evicting an old voice if at the cap).
        2. Run TTS once per (style, out_path) using the style's VoiceSettings.
        3. If we created the voice, delete it before moving to the next ref.
    """
    from io import BytesIO

    setup = _eleven_setup()
    if setup is None:
        return
    client, _retry, print_quota, _is_voice_limit = setup

    from elevenlabs import VoiceSettings
    style_voice_settings = {}
    for style, params in (
        (s, STYLE_PRESETS[s]["elevenlabs"]) for s in jobs_by_style
    ):
        style_voice_settings[style] = VoiceSettings(**params) if params else None

    # Group all jobs by ref_audio. Each group entry is a list of (style, text,
    # out_path); we trust load_jobs to have skipped already-done outputs.
    groups = {}
    for style, jobs in jobs_by_style.items():
        for ref_audio, text, out_path in jobs:
            groups.setdefault(ref_audio, []).append((style, text, out_path))

    print(f"  Modes: all-styles  refs={len(groups)}  "
          f"styles={list(jobs_by_style.keys())}")
    print_quota()

    existing_voices = {}
    try:
        for v in client.voices.get_all().voices:
            existing_voices[v.name] = v.voice_id
        print(f"  Found {len(existing_voices)} existing voices in library")
    except Exception as e:
        print(f"  WARNING: Could not fetch voice library ({e}), will create all voices")

    needed_names = {f"clone_{ref_stem(r)}" for r in groups}
    # Only `clone_*` voices were created by us and are user-deletable.
    # Marketplace voices like "Roger - Laid-Back" come back from get_all() but
    # cannot be deleted, so they must be excluded from eviction candidates.
    eviction_pool = [
        n for n in existing_voices
        if n.startswith("clone_") and n not in needed_names
    ]

    def _evict_one_voice(reason):
        nonlocal eviction_pool
        while eviction_pool:
            name = eviction_pool.pop(0)
            vid = existing_voices.pop(name, None)
            if vid is None:
                continue
            try:
                client.voices.delete(voice_id=vid)
                print(f"    Evicted voice '{name}' to free slot ({reason})")
                return True
            except Exception as e:
                print(f"    WARNING: failed to delete '{name}': {e}")
        # Fallback: any remaining clone_* voice (one that we'd otherwise reuse).
        fallback = [n for n in existing_voices if n.startswith("clone_")]
        if fallback:
            name = fallback[0]
            vid = existing_voices.pop(name)
            try:
                client.voices.delete(voice_id=vid)
                print(f"    Evicted reusable voice '{name}' to free slot ({reason})")
                return True
            except Exception as e:
                print(f"    WARNING: failed to delete '{name}': {e}")
        # Last resort: refetch the library in case our cache is stale.
        try:
            for v in client.voices.get_all().voices:
                existing_voices.setdefault(v.name, v.voice_id)
            fallback = [n for n in existing_voices if n.startswith("clone_")]
            if fallback:
                name = fallback[0]
                vid = existing_voices.pop(name)
                client.voices.delete(voice_id=vid)
                print(f"    Evicted voice '{name}' after refresh ({reason})")
                return True
        except Exception as e:
            print(f"    WARNING: refresh+delete failed: {e}")
        return False

    total_outputs = sum(len(v) for v in groups.values())
    succeeded = 0
    skipped = 0
    voices_created = 0
    voices_deleted = 0
    n_clones = sum(1 for n in existing_voices if n.startswith("clone_"))
    print(f"  Plan: {len(groups)} unique voices to clone, "
          f"{total_outputs} total TTS outputs to generate "
          f"(library has {n_clones} clone_* voices, "
          f"{len(eviction_pool)} evictable)")

    for ref_idx, (ref_audio, items) in enumerate(groups.items(), 1):
        print(f"  [{ref_idx}/{len(groups)}] ref={os.path.basename(ref_audio)}  "
              f"({len(items)} outputs)")
        voice_name = f"clone_{ref_stem(ref_audio)}"
        created_voice_id = None
        try:
            if voice_name in existing_voices:
                voice_id = existing_voices[voice_name]
                print(f"    Reusing existing voice '{voice_name}'")
            else:
                def _create_voice(ra=ref_audio, vn=voice_name):
                    with open(ra, "rb") as f:
                        return client.voices.ivc.create(
                            name=vn, files=[BytesIO(f.read())])
                try:
                    voice = _retry(_create_voice, "create voice")
                except Exception as e:
                    if not _is_voice_limit(e):
                        raise
                    print(f"    voice_limit_reached on create, evicting one voice...")
                    if not _evict_one_voice("voice_limit_reached"):
                        raise RuntimeError(
                            "voice_limit_reached and no voices available to evict")
                    voice = _retry(
                        _create_voice, "create voice (after evict)")
                voice_id = voice.voice_id
                created_voice_id = voice_id
                existing_voices[voice_name] = voice_id
                voices_created += 1
                print(f"    Created new voice '{voice_name}'")
        except Exception as e:
            # Couldn't even make a voice; skip every output for this ref.
            skipped += len(items)
            print(f"    SKIP all {len(items)} outputs (voice error: {e})")
            continue

        try:
            for style, text, out_path in items:
                if os.path.exists(out_path):
                    print(f"      [{style}] already exists, skipping")
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                vs = style_voice_settings.get(style)
                convert_kwargs = dict(
                    voice_id=voice_id,
                    text=text,
                    model_id="eleven_v3",
                    output_format="pcm_24000",
                )
                if vs is not None:
                    convert_kwargs["voice_settings"] = vs
                try:
                    def _tts(kw=convert_kwargs):
                        return b"".join(client.text_to_speech.convert(**kw))
                    pcm_data = _retry(_tts, f"tts/{style}")
                    with wave_mod.open(out_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(pcm_data)
                    succeeded += 1
                    print(f"      [{style}] ✓ {out_path}")
                except Exception as e:
                    skipped += 1
                    print(f"      [{style}] SKIP (error: {e})")
        finally:
            if created_voice_id is not None:
                try:
                    client.voices.delete(voice_id=created_voice_id)
                    existing_voices.pop(voice_name, None)
                    voices_deleted += 1
                except Exception as e:
                    print(f"    WARNING: post-synth delete of '{voice_name}' "
                          f"failed: {e}")

        if ref_idx % 25 == 0:
            print_quota()
            print(f"  Progress: refs={ref_idx}/{len(groups)}  "
                  f"outputs: {succeeded} ok, {skipped} skipped  "
                  f"voices: +{voices_created} -{voices_deleted}")

    print(f"\n  Final: {succeeded} ok, {skipped} skipped of {total_outputs} outputs  "
          f"({voices_created} voices created, {voices_deleted} deleted)")
    print_quota()


RUNNERS = {
    "chatterbox": run_chatterbox,
    "coqui_xtts": run_coqui_xtts,
    "elevenlabs": run_elevenlabs,
}


def _dispatch_model(model_name, args, gpu_id=None):
    """Run a model inside its dedicated venv via subprocess.

    When gpu_id is given, sets CUDA_VISIBLE_DEVICES so the worker only
    sees that single GPU (it will appear as cuda:0 inside the process).
    Returns a (model_name, Popen, log_file) tuple when gpu_id is set
    (non-blocking), or a bool when gpu_id is None (blocking).
    """
    venv_name = VENV_MAP[model_name]
    python = _venv_python(venv_name)
    if not os.path.exists(python):
        print(f"ERROR: Venv '.venv_{venv_name}' not found. Run: bash setup_venvs.sh")
        sys.exit(1)

    cmd = [python, os.path.abspath(__file__), "--model", model_name,
           "--style", ACTIVE_STYLE]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if getattr(args, "all_styles", False) and model_name == "elevenlabs":
        cmd += ["--all-styles"]

    env = {**os.environ, "_GENERATE_SPEECH_WORKER": "1"}
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if gpu_id is not None:
        log_path = os.path.join(OUTPUT_DIR, f"{model_name}.log")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = open(log_path, "w")
        print(f"  Launching {model_name} on GPU {gpu_id} (log: {log_path})")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return model_name, proc, log_file

    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"WARNING: {model_name} failed (exit code {proc.returncode}), continuing...")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloned speech from generation_plan.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Models: " + ", ".join(ALL_MODELS) +
                "\nStyles: " + ", ".join(ALL_STYLES)),
    )
    parser.add_argument(
        "--model", "-m",
        choices=ALL_MODELS + ["all"],
        help="Which model to use (or 'all' to run every model)",
    )
    parser.add_argument(
        "--style", "-s",
        choices=ALL_STYLES,
        default="base",
        help="Style preset controlling expressiveness (default: base)",
    )
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Max number of jobs to run per model (for testing)")
    parser.add_argument("--all-styles", action="store_true",
                        help="ElevenLabs only: clone each voice once and "
                             "generate every style for it before deleting "
                             "the voice. Ignores --style.")
    parser.add_argument("--list", "-l", action="store_true", help="List available models/styles and exit")
    parser.add_argument("--rebuild-plan", action="store_true",
                        help="Rebuild generation plan from validated audio on disk and exit")
    parser.add_argument("--sync", action="store_true",
                        help="Sync the cloned column in the generation plan and exit")
    args = parser.parse_args()

    _init_paths(args.style)

    if args.rebuild_plan:
        _create_plan_from_validated_audio()
        sys.exit(0)

    if args.sync:
        sync_plan()
        sys.exit(0)

    if args.list:
        print("Available models:")
        for m in ALL_MODELS:
            print(f"  {m}")
        print("\nAvailable styles:")
        for s in ALL_STYLES:
            print(f"  {s}")
            for model, params in STYLE_PRESETS[s].items():
                if params:
                    print(f"    {model}: {params}")
        sys.exit(0)

    if not args.model:
        parser.print_help()
        sys.exit(1)

    models = ALL_MODELS if args.model == "all" else [args.model]
    print(f"Style: {ACTIVE_STYLE}  Output: {OUTPUT_DIR}")
    print(f"Plan:  {GENERATION_PLAN_PATH}")

    if args.all_styles and (args.model not in ("elevenlabs", "all")):
        print("ERROR: --all-styles is only supported for --model elevenlabs")
        sys.exit(1)

    if _WORKER_MODE:
        print(f"Device: {DEVICE}")
        if args.all_styles and "elevenlabs" in models:
            jobs_by_style = {}
            for style in ALL_STYLES:
                _init_paths(style)
                style_jobs = load_jobs("elevenlabs")
                if args.limit:
                    style_jobs = style_jobs[:args.limit]
                if style_jobs:
                    jobs_by_style[style] = style_jobs
            if not jobs_by_style:
                print("  Nothing to do across any style.")
            else:
                total = sum(len(v) for v in jobs_by_style.values())
                print(f"\n{'='*50}")
                print(f"  elevenlabs (all-styles) — {total} jobs across "
                      f"{len(jobs_by_style)} styles")
                print(f"{'='*50}")
                run_elevenlabs_all_styles(jobs_by_style)
            for style in jobs_by_style:
                _init_paths(style)
                sync_plan()
            models = [m for m in models if m != "elevenlabs"]
            _init_paths(args.style)

        for model_name in models:
            if model_name not in RUNNERS:
                print(f"  Skipping {model_name}: no runner implemented")
                continue

            jobs = load_jobs(model_name)
            if args.limit:
                jobs = jobs[:args.limit]
            if not jobs:
                print("  Nothing to do.")
                continue

            print(f"\n{'='*50}")
            print(f"  {model_name} — {len(jobs)} jobs")
            print(f"{'='*50}")

            RUNNERS[model_name](jobs)

        if models:
            sync_plan()
    else:
        if len(models) > 1:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"\nLaunching {len(models)} models in parallel across GPUs...")
            handles = []
            gpu_idx = 0
            for model_name in models:
                if model_name == "elevenlabs":
                    print(f"\n  Running elevenlabs (API-based, no GPU)...")
                    _dispatch_model(model_name, args, gpu_id=None)
                else:
                    gpu_id = gpu_idx % 8
                    gpu_idx += 1
                    result = _dispatch_model(model_name, args, gpu_id=gpu_id)
                    handles.append(result)

            if handles:
                print(f"\nWaiting for {len(handles)} GPU models to finish...\n")
                for model_name, proc, log_file in handles:
                    proc.wait()
                    log_file.close()
                    if proc.returncode != 0:
                        print(f"  ✗ {model_name} failed (exit code {proc.returncode})")
                    else:
                        print(f"  ✓ {model_name} finished")
        else:
            _dispatch_model(models[0], args)

        sync_plan()

    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
