"""Upload the voice cloning audio dataset to Hugging Face.

Authenticates with HF (expects ``HF_TOKEN`` env var or prior ``hf auth login``),
then uploads three things to ``kzhou/voice_cloning_style_transfer``:

    README.md     dataset card (this directory)
    LICENSE.txt   CC-BY-NC 4.0 + forbidden uses
    audio_data/*  the 4 audio splits (17 GB, 55,684 wavs)
    metadata/*    ratings, demographics, accent CSVs

Run:
    python hf_dataset/upload.py --dry-run    # show what would be uploaded
    python hf_dataset/upload.py              # actual upload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID  = "kzhou/voice_cloning_style_transfer"
REPO_TYPE = "dataset"

HERE   = Path(__file__).resolve().parent
PUBLIC = HERE.parent
AUDIO  = PUBLIC / "audio_data"
DATA   = PUBLIC / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be uploaded and exit without contacting HF.")
    ap.add_argument("--only", choices=["card", "metadata", "original", "cloned",
                                       "cloned_styles", "cloned_iterative", "all"],
                    default="all", help="Upload just one category (useful for retries).")
    args = ap.parse_args()

    api = HfApi()

    # ── Verify auth ────────────────────────────────────────────────────────
    if not args.dry_run:
        try:
            user = api.whoami()
            print(f"Authenticated as: {user['name']}")
        except Exception as e:
            print("Not authenticated. Set HF_TOKEN or run 'hf auth login'.", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            sys.exit(2)
        print(f"Target repo: {REPO_ID} ({REPO_TYPE})")
        # Make sure the repo exists (won't error if it already does)
        api.create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    # ── Pre-upload sanity check: no Prolific IDs leaking through ──────────
    import re
    PROLIFIC = re.compile(r"(?<![a-f0-9])[a-f0-9]{24}(?![a-f0-9])")
    print("\n=== sanity: no raw Prolific IDs in any filename being uploaded ===")
    leak = 0
    for d in [AUDIO, DATA]:
        for p in d.rglob("*"):
            if p.is_file() and PROLIFIC.search(p.name):
                print(f"  LEAK: {p}")
                leak += 1
    if leak:
        print(f"\n  ABORT: {leak} files with raw Prolific IDs in the name")
        sys.exit(3)
    print("  → 0 leaks")

    # ── Plan: a list of (local_path, path_in_repo, label) tuples ──────────
    plans = []
    if args.only in ("card", "all"):
        plans.append((HERE / "README.md",     "README.md",     "dataset card"))
        plans.append((HERE / "LICENSE.txt",   "LICENSE.txt",   "license"))
    if args.only in ("metadata", "all"):
        for csv in sorted(DATA.glob("*.csv")):
            plans.append((csv, f"metadata/{csv.name}", "metadata CSV"))
    if args.only in ("original", "all"):
        plans.append((AUDIO / "original",         "original",         "1.4 GB"))
    if args.only in ("cloned", "all"):
        plans.append((AUDIO / "cloned",           "cloned",           "551 MB"))
    if args.only in ("cloned_styles", "all"):
        plans.append((AUDIO / "cloned_styles",    "cloned_styles",    "3.4 GB"))
    if args.only in ("cloned_iterative", "all"):
        plans.append((AUDIO / "cloned_iterative", "cloned_iterative", "12 GB"))

    print("\n=== upload plan ===")
    total_files = total_bytes = 0
    for local, remote, label in plans:
        if local.is_file():
            n_files, n_bytes = 1, local.stat().st_size
        else:
            n_files = sum(1 for _ in local.rglob("*") if _.is_file())
            n_bytes = sum(_.stat().st_size for _ in local.rglob("*") if _.is_file())
        total_files += n_files; total_bytes += n_bytes
        print(f"  {label:>14s}  →  {remote:<24}  {n_files:>6} files  {n_bytes/1e9:>5.2f} GB  (from {local.relative_to(PUBLIC)})")
    print(f"  ─────")
    print(f"  TOTAL                                       {total_files:>6} files  {total_bytes/1e9:>5.2f} GB")

    if args.dry_run:
        print("\n(dry run — nothing uploaded)")
        return

    # ── Actual upload ──────────────────────────────────────────────────────
    for local, remote, label in plans:
        print(f"\n→ uploading {label}: {local} → {REPO_ID}:{remote}")
        if local.is_file():
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Add {label}",
            )
        else:
            # upload_folder handles large directories with chunking + retries
            api.upload_folder(
                folder_path=str(local),
                path_in_repo=remote,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Add {label}",
                allow_patterns=["*.wav", "*.csv", "*.md", "*.txt"],
            )

    print("\n=== upload complete ===")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
