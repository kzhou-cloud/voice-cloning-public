#!/usr/bin/env python3
"""
Iterative clone-of-clone experiment.

Takes cloned files from tts_outputs/{model}/ (skipping _long) and re-clones
9 rounds. Each round reads the previous round's output, advances to the next
Grandfather Passage sentence (wrapping 9->1), and saves to
tts_outputs_{round}/{model}/. Writes a generation_plan.csv per round.

Usage:
    python generate_iterative_clones.py -m chatterbox --complete-only --dry-run
    python generate_iterative_clones.py -m coqui_xtts
    python generate_iterative_clones.py -m chatterbox --rounds 2 5
"""

import argparse
import csv
import os
import re
import sys
import warnings
import wave as wave_mod
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
VOICE_DIR = os.path.join(REPO_ROOT, "voice_cloning")
MODELS = ["chatterbox", "coqui_xtts"]
SENTENCES = [
    "You wish to know about my grandfather.",
    "Well, he is nearly 93 years old, yet he still thinks as swiftly as ever.",
    "He dresses himself in an old black frock coat, usually several buttons missing.",
    "A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect.",
    "When he speaks, his voice is just a bit cracked and quivers a bit.",
    "Twice each day he plays skillfully and with zest upon a small organ.",
    "Except in the winter when the snow or ice prevents, he slowly takes a short walk in the open air each day.",
    'We have often urged him to walk more and smoke less, but he always answers, "Banana oil!"',
    "Grandfather likes to be modern in his language.",
]
CSV_FIELDS = [
    "model_name", "participant_id", "round", "start_sentence",
    "input_clip", "input_text", "target_text",
    "predicted_output_clip", "missing_input_clip", "cloned",
]


def next_sent(s):
    return s % 9 + 1


def parse_sent(filename):
    m = re.search(r"_sentence_(\d+)", filename)
    return int(m.group(1)) if m else None


def out_name(model, pid, start, rnd):
    tgt = (start + rnd - 2) % 9 + 1
    return f"cloned_{model}_{pid}_sentence_{tgt}_from{start}_round{rnd}.wav"


def out_dir(model, rnd):
    return os.path.join(ROOT, f"tts_outputs_{rnd}", model)


def inp_path(model, pid, start, rnd):
    if rnd == 2:
        return os.path.join(VOICE_DIR, "tts_outputs", model,
                            f"cloned_{model}_{pid}_sentence_{start}.wav")
    return os.path.join(out_dir(model, rnd - 1), out_name(model, pid, start, rnd - 1))


def discover(model, complete_only=False):
    r1 = os.path.join(VOICE_DIR, "tts_outputs", model)
    if not os.path.isdir(r1):
        sys.exit(f"ERROR: {r1} not found")
    prefix = f"cloned_{model}_"
    pid_sents = defaultdict(set)
    for f in sorted(os.listdir(r1)):
        if not f.endswith(".wav") or "_long" in f or not f.startswith(prefix):
            continue
        s = parse_sent(f)
        m = re.match(re.escape(prefix) + r"(.+)_sentence_\d+", f)
        if s and m:
            pid_sents[m.group(1)].add(s)
    if complete_only:
        pid_sents = {p: ss for p, ss in pid_sents.items() if ss == set(range(1, 10))}
    return [(p, s) for p in sorted(pid_sents) for s in sorted(pid_sents[p])]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# -- model loaders ----------------------------------------------------------

def _get_device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_chatterbox():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    device = _get_device()
    print(f"Device: {device}\nLoading ChatterboxTTS...")
    model = ChatterboxTTS.from_pretrained(device=device)
    def generate(text, ref, out):
        wav = model.generate(text, audio_prompt_path=ref)
        ta.save(out, wav, model.sr, encoding="PCM_F", bits_per_sample=32)
    return generate


def load_coqui_xtts():
    import numpy as np, soundfile as sf, torch, torchaudio as ta
    from TTS.api import TTS as CoquiTTS
    os.environ["COQUI_TOS_AGREED"] = "1"
    device = _get_device()
    orig_load, orig_ta = torch.load, ta.load
    def patched(fp, *a, **kw):
        d, sr = sf.read(fp)
        return __import__("torch").from_numpy((d.reshape(1,-1) if d.ndim==1 else d.T).astype(np.float32)), sr
    torch.load = lambda *a, **kw: orig_load(*a, **{**kw, "weights_only": False})
    ta.load = patched
    print(f"Device: {device}\nLoading XTTS-v2...")
    tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2"); tts.to(device)
    torch.load = orig_load
    def generate(text, ref, out):
        tts.tts_to_file(text=text, file_path=out, speaker_wav=ref, language="en")
    return generate


LOADERS = {"chatterbox": load_chatterbox, "coqui_xtts": load_coqui_xtts}


# -- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Iterative clone-of-clone experiment")
    p.add_argument("--model", "-m", required=True, choices=MODELS)
    p.add_argument("--rounds", nargs=2, type=int, default=[2, 11],
                   metavar=("START", "END"), help="[start, end) range (default: 2 11)")
    p.add_argument("--complete-only", action="store_true",
                   help="Only participants with all 9 sentences in round 1")
    p.add_argument("--chunk", nargs=2, type=int, default=None,
                   metavar=("IDX", "TOTAL"),
                   help="Process chunk IDX of TOTAL (0-indexed) for parallel runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    model, (rstart, rend) = args.model, args.rounds

    chains = discover(model, complete_only=args.complete_only)
    if args.chunk:
        idx, total = args.chunk
        pids = sorted(set(p for p, _ in chains))
        chunk_size = (len(pids) + total - 1) // total
        my_pids = set(pids[idx * chunk_size : (idx + 1) * chunk_size])
        chains = [(p, s) for p, s in chains if p in my_pids]
    print(f"{len(chains)} chains from {len(set(p for p,_ in chains))} participants ({model})")

    # Build plan
    csv_rows, pending = {}, []
    for r in range(rstart, rend):
        csv_rows[r] = []
        for pid, start in chains:
            src = inp_path(model, pid, start, r)
            dst = os.path.join(out_dir(model, r), out_name(model, pid, start, r))
            in_s = parse_sent(os.path.basename(src))
            tgt = next_sent(in_s)
            row = {
                "model_name": model, "participant_id": pid, "round": r,
                "start_sentence": start,
                "input_clip": os.path.relpath(src, ROOT),
                "input_text": SENTENCES[in_s - 1],
                "target_text": SENTENCES[tgt - 1],
                "predicted_output_clip": os.path.relpath(dst, ROOT),
                "missing_input_clip": int(not os.path.exists(src)),
                "cloned": int(os.path.exists(dst)),
            }
            csv_rows[r].append(row)
            if not row["cloned"]:
                pending.append((r, pid, start, src, dst, in_s, tgt))

    skipped = len(chains) * (rend - rstart) - len(pending)
    counts = Counter(r for r, *_ in pending)
    print(f"\nTotal: {len(pending)} jobs, {skipped} already done")
    for r in range(rstart, rend):
        print(f"  Round {r:2d}: {counts.get(r,0)} to generate, "
              f"{len(chains) - counts.get(r,0)} already done")

    # Write CSVs
    for r in range(rstart, rend):
        d = out_dir(model, r)
        os.makedirs(d, exist_ok=True)
        write_csv(os.path.join(d, "generation_plan.csv"), csv_rows[r])

    if args.dry_run:
        print(f"\n{'='*60}\n  Execution plan\n{'='*60}")
        for i, (r, pid, start, src, dst, in_s, tgt) in enumerate(pending, 1):
            ok = "ok" if os.path.exists(src) else "MISSING"
            print(f"  {i:4d}. round={r} pid={pid[:16]}... "
                  f"sent {in_s}->{tgt} from{start} src={ok} -> {os.path.basename(dst)}")
        return

    if not pending:
        return print("Nothing to do.")

    generate = LOADERS[model]()
    for i, (r, pid, start, src, dst, in_s, tgt) in enumerate(pending, 1):
        if i == 1 or pending[i-2][0] != r:
            print(f"\n{'='*60}\n  Round {r}\n{'='*60}")
        if not os.path.exists(src):
            print(f"  SKIP {pid[:16]}... round {r}: input missing"); continue
        print(f"  [{i}/{len(pending)}] round={r} pid={pid[:16]}... sent {in_s}->{tgt}")
        try:
            generate(SENTENCES[tgt - 1], src, dst)
        except Exception as e:
            print(f"    ERROR: {e}")
            if os.path.exists(dst): os.remove(dst)

    # Update CSVs with final cloned status
    for r in range(rstart, rend):
        for row in csv_rows[r]:
            row["cloned"] = int(os.path.exists(os.path.join(ROOT, row["predicted_output_clip"])))
        write_csv(os.path.join(out_dir(model, r), "generation_plan.csv"), csv_rows[r])
    print(f"\nDone. {len(pending)} clips generated.")


if __name__ == "__main__":
    main()
