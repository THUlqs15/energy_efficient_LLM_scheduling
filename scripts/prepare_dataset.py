"""Synthesise trace.jsonl from ShareGPT52K dataset.

Usage:  python scripts/prepare_dataset.py

Edit the USER KNOBS section below to change parameters.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download


# ==============================================================================
# USER KNOBS — change these to control trace generation
# ==============================================================================
OUTPUT = "trace.jsonl"
# Path of the generated JSONL file (one request per line).

NUM_REQUESTS = 400
# Number of requests to sample into the trace.

RATE_QPS = 10
# Arrival rate in requests/second. Record i arrives at time i / RATE_QPS.

TTFT_MEAN_MS = 1000.0
# Mean TTFT SLO requirement (ms). Sampled from a truncated normal distribution.

TTFT_STD_MS = 200.0
# Std dev of the TTFT SLO requirement (ms).

TPOT_MEAN_MS = 50.0
# Mean TPOT SLO requirement (ms).

TPOT_STD_MS = 20.0
# Std dev of the TPOT SLO requirement (ms).

MIN_OUTPUT_TOKENS = 64
# Minimum number of output tokens per request (uniformly sampled).

MAX_OUTPUT_TOKENS = 1024
# Maximum number of output tokens per request (uniformly sampled).

MIN_PROMPT_CHARS = 64
# Minimum prompt length in characters (filter out very short prompts).

MAX_PROMPT_CHARS = 8000
# Maximum prompt length in characters (filter out very long prompts).

SEED = 42
# Random seed for reproducibility (shuffling, SLO sampling, token counts).

DATASET_DIR = "data/sharegpt52k"
# Local path to the ShareGPT52K dataset (downloaded once via huggingface_hub).

REPO_ID = "RyokoAI/ShareGPT52K"
# Hugging Face repository ID (re-downloaded if Git LFS pointers detected).
# ==============================================================================


def _ensure_dataset():
    ds_dir = Path(DATASET_DIR)
    if not ds_dir.is_dir():
        print(f"[prepare_dataset] Dataset not found — downloading from {REPO_ID} ...")
        snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                          local_dir=DATASET_DIR, local_dir_use_symlinks=False)
        return

    # Check for Git LFS pointer files (134 bytes, starts with "version")
    for fpath in ds_dir.glob("*.json"):
        if fpath.stat().st_size < 200:
            with open(fpath) as f:
                header = f.read(40)
            if header.startswith("version "):
                print(f"[prepare_dataset] Git LFS pointers detected — "
                      f"re-downloading dataset ...")
                shutil.rmtree(ds_dir)
                snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                                  local_dir=DATASET_DIR, local_dir_use_symlinks=False)
                return


def truncated_normal(mean: float, std: float, low: float = 1.0) -> float:
    while True:
        v = random.gauss(mean, std)
        if v > low:
            return v


def main():
    random.seed(SEED)

    _ensure_dataset()

    # Load dataset
    candidates = []
    ds_dir = Path(DATASET_DIR)
    for fpath in sorted(ds_dir.glob("*.json")):
        with open(fpath) as f:
            convos = json.load(f)
        for convo in convos:
            messages = convo.get("conversations", convo.get("messages", []))
            for msg in messages:
                role = msg.get("from", msg.get("role", ""))
                content = msg.get("value", msg.get("content", ""))
                if role.lower() in ("human", "user"):
                    if MIN_PROMPT_CHARS <= len(content) <= MAX_PROMPT_CHARS:
                        candidates.append(content)
                    break

    print(f"[prepare_dataset] Found {len(candidates)} eligible prompts")
    num = NUM_REQUESTS
    if len(candidates) < num:
        print(f"[prepare_dataset] WARNING: not enough prompts; using all {len(candidates)}")
        num = len(candidates)

    random.shuffle(candidates)
    prompts = candidates[:num]

    # Write trace
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for i, prompt in enumerate(prompts):
            arrival_s = i / RATE_QPS
            max_tokens = random.randint(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
            ttft_ms = truncated_normal(TTFT_MEAN_MS, TTFT_STD_MS)
            tpot_ms = truncated_normal(TPOT_MEAN_MS, TPOT_STD_MS)
            record = {
                "id": f"req_{i:06d}",
                "arrival_s": round(arrival_s, 6),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ttft_ms": round(ttft_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
                "w_n": 1.0,
            }
            f.write(json.dumps(record) + "\n")

    print(f"[prepare_dataset] Wrote {num} records to {OUTPUT}")


if __name__ == "__main__":
    main()
