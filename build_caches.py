#!/usr/bin/env python3
"""Rebuild every audio-feature cache that the analysis notebooks read.

Outputs under ``cache/``:
    audio_identity_features_cache.csv     librosa MFCC + spectral + RMS + zcr per wav
    acoustic_stats_chatterbox.csv         librosa F0 + duration (chatterbox iterative)
    emotion_stats_chatterbox.csv          NVIDIA Audio2Emotion-v3 per wav
    embeddings_cache_chatterbox_ecapa.npz SpeechBrain ECAPA-TDNN 192-D embeddings
    style_analysis_cache_chatterbox.csv   librosa features for style-variant wavs

Usage:
    # full rebuild on this machine (uses all GPUs visible to CUDA + all CPUs)
    python build_caches.py

    # one cache at a time
    python build_caches.py --cache emotion
    python build_caches.py --cache ecapa
    python build_caches.py --cache librosa     # all three CPU caches

    # single-shard worker (called internally by the launcher; usually you don't run this directly)
    CUDA_VISIBLE_DEVICES=3 python build_caches.py --cache emotion --shard 3/8

Layout assumed:
    audio_data/original/preprocessed_sentences/sentence_N_valid/speaker_NNN_sentence_N.wav
    audio_data/cloned/{model}/cloned_{model}_speaker_NNN_sentence_N.wav
    audio_data/cloned_styles/tts_outputs_{style}/{model}/cloned_{model}_speaker_NNN_sentence_N.wav
    audio_data/cloned_iterative/tts_outputs_{R}/{model}/cloned_{model}_speaker_NNN_sentence_N_from{F}_round{R}.wav
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet down third-party noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE      = Path(__file__).resolve().parent
AUDIO     = HERE / "audio_data"
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL = "chatterbox"       # the model the paper analyzes for ECAPA / emotion / acoustic
MAX_ROUND = 50
TARGET_SR = 16_000


# ──────────────────────────────────────────────────────────────────────────────
# File-list builders
# ──────────────────────────────────────────────────────────────────────────────

PROLIFIC_OR_ANON = r"(speaker_\d+|[a-f0-9]{24})"
ORIG_RE   = re.compile(rf"^{PROLIFIC_OR_ANON}_sentence_(\d+)\.wav$")
CLONE_RE  = re.compile(rf"^cloned_(?P<model>[a-z_]+?)_{PROLIFIC_OR_ANON}_sentence_(\d+)\.wav$")
ITER_RE   = re.compile(rf"^cloned_(?P<model>[a-z_]+?)_{PROLIFIC_OR_ANON}_sentence_(\d+)_from(\d+)_round(\d+)\.wav$")


def discover_files_iterative(model: str = MODEL):
    """Files needed for the iterative-cloning analysis: round 0 (originals),
    round 1 (cloned/), and rounds 2..50 (cloned_iterative/)."""
    rows = []

    # round 0: originals
    src = AUDIO / "original" / "preprocessed_sentences"
    for sent in range(1, 10):
        d = src / f"sentence_{sent}_valid"
        if not d.is_dir(): continue
        for f in sorted(d.iterdir()):
            m = ORIG_RE.match(f.name)
            if m:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(1),
                             "sentence": int(m.group(2)),
                             "round": 0})

    # round 1: cloned/
    d = AUDIO / "cloned" / model
    if d.is_dir():
        for f in sorted(d.iterdir()):
            m = CLONE_RE.match(f.name)
            if m and m.group("model") == model:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(2),
                             "sentence": int(m.group(3)),
                             "round": 1})

    # rounds 2..50: cloned_iterative/tts_outputs_R/<model>/
    for r in range(2, MAX_ROUND + 1):
        d = AUDIO / "cloned_iterative" / f"tts_outputs_{r}" / model
        if not d.is_dir(): continue
        for f in sorted(d.iterdir()):
            m = ITER_RE.match(f.name)
            if m and m.group("model") == model:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(2),
                             "sentence": int(m.group(3)),
                             "round": int(m.group(5))})
    return pd.DataFrame(rows)


def discover_files_audio_identity():
    """Files needed for the speaker-identity probe (Figure 17).
    For each base model: originals + step-0 clones."""
    rows = []
    # originals (one row per (speaker, sentence))
    src = AUDIO / "original" / "preprocessed_sentences"
    for sent in range(1, 10):
        d = src / f"sentence_{sent}_valid"
        if not d.is_dir(): continue
        for f in sorted(d.iterdir()):
            m = ORIG_RE.match(f.name)
            if m:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(1),
                             "sentence": int(m.group(2)),
                             "model": "original",
                             "is_cloned": False,
                             "voice_type": "original"})
    # cloned/{model}/
    for sub in (AUDIO / "cloned").iterdir() if (AUDIO / "cloned").is_dir() else []:
        if not sub.is_dir(): continue
        model = sub.name
        for f in sorted(sub.iterdir()):
            m = CLONE_RE.match(f.name)
            if m and m.group("model") == model:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(2),
                             "sentence": int(m.group(3)),
                             "model": model,
                             "is_cloned": True,
                             "voice_type": "cloned"})
    return pd.DataFrame(rows)


def discover_files_styles(model: str = MODEL):
    """Files needed for Figure 10 (chatter_box_style_embeddings): the source
    recordings (style='original'), the default-settings step-0 clones
    (style='base'), and every cloned_styles variant for the given model."""
    rows = []

    # style="original" — source recordings
    src = AUDIO / "original" / "preprocessed_sentences"
    for sent in range(1, 10):
        d = src / f"sentence_{sent}_valid"
        if not d.is_dir(): continue
        for f in sorted(d.iterdir()):
            m = ORIG_RE.match(f.name)
            if m:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(1),
                             "sentence": int(m.group(2)),
                             "model": model,
                             "style": "original"})

    # style="base" — default-settings clones
    d = AUDIO / "cloned" / model
    if d.is_dir():
        for f in sorted(d.iterdir()):
            m = CLONE_RE.match(f.name)
            if m and m.group("model") == model:
                rows.append({"path": str(f.relative_to(HERE)),
                             "speaker_id": m.group(2),
                             "sentence": int(m.group(3)),
                             "model": model,
                             "style": "base"})

    # style-variant clones
    root = AUDIO / "cloned_styles"
    if root.is_dir():
        for style_dir in sorted(root.iterdir()):
            if not style_dir.is_dir() or not style_dir.name.startswith("tts_outputs_"):
                continue
            style = style_dir.name[len("tts_outputs_"):]
            d = style_dir / model
            if not d.is_dir(): continue
            for f in sorted(d.iterdir()):
                m = CLONE_RE.match(f.name)
                if m and m.group("model") == model:
                    rows.append({"path": str(f.relative_to(HERE)),
                                 "speaker_id": m.group(2),
                                 "sentence": int(m.group(3)),
                                 "model": model,
                                 "style": style})
    return pd.DataFrame(rows)


def shard_rows(df: pd.DataFrame, shard: str) -> pd.DataFrame:
    """``shard`` is 'N/M'; returns the N-th of M evenly split chunks (0-indexed)."""
    n, m = map(int, shard.split("/"))
    sub = df.iloc[n::m].reset_index(drop=True)
    return sub


# ──────────────────────────────────────────────────────────────────────────────
# Cache 1: librosa MFCC + spectral identity features (CPU, multiproc)
# ──────────────────────────────────────────────────────────────────────────────

def _librosa_identity_row(row_dict):
    """Matches the paper's extract_audio_features() exactly:
    14 acoustic features + 26 MFCC features = 40-D vector per clip."""
    import librosa
    p = row_dict["path"]
    try:
        y, sr = librosa.load(p, sr=TARGET_SR)
        if len(y) == 0:
            raise ValueError(f"Empty audio: {p}")

        rms       = librosa.feature.rms(y=y)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
        centroid  = librosa.feature.spectral_centroid (y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff   = librosa.feature.spectral_rolloff  (y=y, sr=sr)[0]
        mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        feats = {
            "duration_sec":    len(y) / sr,
            "rms_mean":        float(np.sqrt(np.mean(y ** 2))),
            "rms_std":         float(np.std(np.abs(y))),
            "zcr_mean":        float(np.mean(np.abs(np.diff(np.signbit(y))).astype(float))),
            "rms_frame_mean":  float(rms.mean()),
            "rms_frame_std":   float(rms.std()),
            "zcr_frame_mean":  float(zcr.mean()),
            "zcr_frame_std":   float(zcr.std()),
            "centroid_mean":   float(centroid.mean()),
            "centroid_std":    float(centroid.std()),
            "bandwidth_mean":  float(bandwidth.mean()),
            "bandwidth_std":   float(bandwidth.std()),
            "rolloff_mean":    float(rolloff.mean()),
            "rolloff_std":     float(rolloff.std()),
        }
        for i in range(mfcc.shape[0]):
            feats[f"mfcc{i+1:02d}_mean"] = float(mfcc[i].mean())
            feats[f"mfcc{i+1:02d}_std"]  = float(mfcc[i].std())
    except Exception as e:
        feats = {"error": str(e)}
    out = dict(row_dict); out.update(feats); return out


def build_audio_identity_cache(out_path: Path):
    from multiprocessing import Pool, cpu_count
    df = discover_files_audio_identity()
    df["local_path"] = df["path"]
    print(f"  audio_identity: {len(df):,} wavs, using {cpu_count()} CPUs")
    rows = df.to_dict("records")
    with Pool(cpu_count()) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_librosa_identity_row, rows, chunksize=32)):
            results.append(r)
            if i and i % 500 == 0:
                print(f"    {i:>6}/{len(rows)}")
    out = pd.DataFrame(results)
    # Order columns: metadata first, then features
    meta = ["speaker_id", "sentence", "model", "is_cloned", "voice_type", "local_path"]
    feat = [c for c in out.columns if c not in meta and c != "path"]
    out = out[meta + feat]
    # Match historical column name `sentence_num` for compatibility with the
    # existing analysis notebook
    out = out.rename(columns={"sentence": "sentence_num"})
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out):,} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Cache 2: acoustic stats (librosa pyin + duration) — CPU
# ──────────────────────────────────────────────────────────────────────────────

def _acoustic_row(path):
    import librosa
    try:
        y, sr = librosa.load(path, sr=TARGET_SR)
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=1024)
        voiced_f0 = f0[voiced]
        med = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else np.nan
        return {"path": path, "duration_s": len(y) / sr, "median_f0_hz": med}
    except Exception as e:
        return {"path": path, "duration_s": np.nan, "median_f0_hz": np.nan, "error": str(e)}


def build_acoustic_stats(out_path: Path):
    from multiprocessing import Pool, cpu_count
    df = discover_files_iterative(MODEL)
    print(f"  acoustic_stats: {len(df):,} wavs, using {cpu_count()} CPUs")
    paths = df["path"].tolist()
    with Pool(cpu_count()) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_acoustic_row, paths, chunksize=32)):
            results.append(r)
            if i and i % 1000 == 0:
                print(f"    {i:>6}/{len(paths)}")
    out = pd.DataFrame(results)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out):,} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Cache 3: style-variant librosa features
# ──────────────────────────────────────────────────────────────────────────────

def _style_row(row_dict):
    import librosa
    p = row_dict["path"]
    try:
        y, sr = librosa.load(p, sr=TARGET_SR)
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, hop_length=1024)
        voiced_f0 = f0[voiced]
        med_f0  = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else np.nan
        std_f0  = float(np.std (voiced_f0)) if len(voiced_f0) > 0 else np.nan
        # Spectral tilt = slope of mean log-power vs frequency (paper convention)
        S = np.abs(librosa.stft(y, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr)
        mean_log_power = np.log10(S + 1e-10).mean(axis=1)
        spectral_tilt = float(np.polyfit(freqs, mean_log_power, 1)[0])
        feats = {
            "duration_s": len(y) / sr,
            "median_f0_hz": med_f0,
            "f0_std_hz":    std_f0,
            "rms_mean":     float(np.mean(librosa.feature.rms(y=y))),
            "rms_std":      float(np.std (librosa.feature.rms(y=y))),
            "zcr_mean":     float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "spectral_centroid_mean":  float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
            "spectral_bandwidth_mean": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
            "spectral_tilt": spectral_tilt,
        }
    except Exception as e:
        feats = {"error": str(e)}
    out = dict(row_dict); out.update(feats); return out


def build_style_cache(out_path: Path):
    from multiprocessing import Pool, cpu_count
    df = discover_files_styles(MODEL)
    print(f"  style_analysis: {len(df):,} wavs, using {cpu_count()} CPUs")
    if df.empty:
        print("    no style wavs found — skipping")
        return
    rows = df.to_dict("records")
    with Pool(cpu_count()) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_style_row, rows, chunksize=32)):
            results.append(r)
            if i and i % 500 == 0:
                print(f"    {i:>6}/{len(rows)}")
    out = pd.DataFrame(results)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out):,} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Cache 4: ECAPA-TDNN embeddings (sharded across 8 GPUs)
# ──────────────────────────────────────────────────────────────────────────────

def build_ecapa_shard(shard: str, out_npz: Path):
    import torch, torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = discover_files_iterative(MODEL)
    df = shard_rows(df, shard)
    if df.empty:
        print(f"  ecapa shard {shard}: empty"); return

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    embs = []
    paths = df["path"].tolist()
    for p in tqdm(paths, desc=f"ecapa-{shard}", mininterval=10.0):
        try:
            wav, sr = torchaudio.load(p)
            if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
            with torch.no_grad():
                e = enc.encode_batch(wav.to(device)).squeeze().cpu().numpy()
        except Exception as exc:
            e = np.full(192, np.nan)
        embs.append(e)

    np.savez_compressed(out_npz, embeddings=np.array(embs), paths=np.array(paths))
    print(f"  wrote {out_npz}  ({len(embs)} embeddings, shard {shard})")


def merge_ecapa_shards(out_npz: Path, n_shards: int):
    paths_all, emb_all = [], []
    for i in range(n_shards):
        f = out_npz.with_name(out_npz.stem + f".shard{i}.npz")
        d = np.load(f, allow_pickle=True)
        paths_all.extend(d["paths"].tolist())
        emb_all.append(d["embeddings"])
        f.unlink()
    np.savez_compressed(out_npz,
                        embeddings=np.vstack(emb_all),
                        paths=np.array(paths_all))
    print(f"  merged {n_shards} shards → {out_npz}  ({sum(len(p) for p in paths_all if isinstance(p, list)) if False else len(paths_all)} embeddings)")


# ──────────────────────────────────────────────────────────────────────────────
# Cache 5: NVIDIA Audio2Emotion-v3 (sharded across 8 GPUs)
# ──────────────────────────────────────────────────────────────────────────────

def build_emotion_shard(shard: str, out_csv: Path):
    import ctypes
    # Pre-load cuDNN so onnxruntime-gpu can find it
    for candidate in [
        "/home/kzhou/voice_cloning_root/voice_cloning/.venv_main/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
        "/home/kzhou/voice_cloning_root/voice_cloning/.venv_main/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so",
    ]:
        if Path(candidate).exists():
            try: ctypes.CDLL(candidate)
            except OSError: pass
            break
    import librosa
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from scipy.special import softmax
    from tqdm import tqdm

    model_path = hf_hub_download("nvidia/Audio2Emotion-v3.0", "network.onnx")
    sess = ort.InferenceSession(model_path,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

    df = discover_files_iterative(MODEL)
    df = shard_rows(df, shard)
    if df.empty:
        print(f"  emotion shard {shard}: empty"); return

    rows = []
    for p in tqdm(df["path"].tolist(), desc=f"emotion-{shard}", mininterval=10.0):
        try:
            y, _ = librosa.load(p, sr=TARGET_SR)
            rem = len(y) % 5000
            if rem != 0:
                y = np.pad(y, (0, 5000 - rem))
            inp = y.reshape(1, -1).astype(np.float32)
            logits = sess.run(None, {"input_values": inp})[0][0]
            probs = softmax(logits)
            r = {f"emo_{e}": float(p_) for e, p_ in zip(emotions, probs)}
        except Exception as e:
            r = {f"emo_{e}": np.nan for e in emotions}
            r["error"] = str(e)
        r["path"] = p
        rows.append(r)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({len(rows)} rows, shard {shard})")


def merge_emotion_shards(out_csv: Path, n_shards: int):
    parts = []
    for i in range(n_shards):
        f = out_csv.with_name(out_csv.stem + f".shard{i}.csv")
        parts.append(pd.read_csv(f))
        f.unlink()
    pd.concat(parts, ignore_index=True).to_csv(out_csv, index=False)
    print(f"  merged {n_shards} shards → {out_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# Launcher: spread N shards across the visible GPUs as subprocesses
# ──────────────────────────────────────────────────────────────────────────────

def detect_n_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True
        ).strip().splitlines()
        return len(out)
    except Exception:
        return 0


def parallel_gpu(cache_name: str, n_shards: int, out_base: Path, suffix: str):
    """Launch n_shards subprocesses, each on its own GPU, each computing one shard."""
    procs = []
    log_dir = HERE / "cache" / "_build_logs"
    log_dir.mkdir(exist_ok=True)
    for i in range(n_shards):
        out_path = out_base.with_name(out_base.stem + f".shard{i}{suffix}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        log = open(log_dir / f"{cache_name}.shard{i}.log", "w")
        p = subprocess.Popen(
            [sys.executable, __file__, "--cache", cache_name,
             "--shard", f"{i}/{n_shards}", "--out", str(out_path)],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
        )
        procs.append((p, log, i))
        print(f"  launched {cache_name} shard {i}/{n_shards} on GPU {i}, log={log.name}")
    for p, log, i in procs:
        ret = p.wait()
        log.close()
        print(f"  shard {i} exited with code {ret}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    choices=["all", "librosa", "ecapa", "emotion",
                             "identity", "acoustic", "style"],
                    default="all")
    ap.add_argument("--shard", default=None, help="N/M shard for the parallel workers")
    ap.add_argument("--out", default=None, help="output path override (for sharded calls)")
    args = ap.parse_args()

    # ── Sharded worker mode: called by the launcher ────────────────────────
    if args.shard:
        out = Path(args.out) if args.out else None
        if args.cache == "ecapa":
            build_ecapa_shard(args.shard, out)
        elif args.cache == "emotion":
            build_emotion_shard(args.shard, out)
        else:
            print(f"--shard is only valid for cache=ecapa|emotion", file=sys.stderr)
            sys.exit(2)
        sys.exit(0)

    # ── Top-level driver ───────────────────────────────────────────────────
    n_gpu = detect_n_gpus()
    print(f"=== build_caches.py ===")
    print(f"  cache dir: {CACHE_DIR}")
    print(f"  audio dir: {AUDIO}")
    print(f"  GPUs:      {n_gpu}")
    print()

    if args.cache in ("all", "librosa", "identity"):
        print("[1] audio_identity_features (librosa, CPU)")
        build_audio_identity_cache(CACHE_DIR / "audio_identity_features_cache.csv")
    if args.cache in ("all", "librosa", "acoustic"):
        print("[2] acoustic_stats (librosa pyin, CPU)")
        build_acoustic_stats(CACHE_DIR / f"acoustic_stats_{MODEL}.csv")
    if args.cache in ("all", "librosa", "style"):
        print("[3] style_analysis_cache (librosa, CPU)")
        build_style_cache(CACHE_DIR / f"style_analysis_cache_{MODEL}.csv")
    if args.cache in ("all", "ecapa") and n_gpu > 0:
        print(f"[4] ECAPA embeddings (sharded across {n_gpu} GPUs)")
        out = CACHE_DIR / f"embeddings_cache_{MODEL}_ecapa.npz"
        parallel_gpu("ecapa", n_gpu, out, ".npz")
        merge_ecapa_shards(out, n_gpu)
    if args.cache in ("all", "emotion") and n_gpu > 0:
        print(f"[5] Audio2Emotion (sharded across {n_gpu} GPUs)")
        out = CACHE_DIR / f"emotion_stats_{MODEL}.csv"
        parallel_gpu("emotion", n_gpu, out, ".csv")
        merge_emotion_shards(out, n_gpu)

    print("\n=== done ===")
