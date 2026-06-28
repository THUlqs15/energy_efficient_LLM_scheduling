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
from transformers import AutoTokenizer


# ==============================================================================
# USER KNOBS — change these to control trace generation
# ==============================================================================
OUTPUT = "trace.jsonl"
# Path of the generated JSONL file (one request per line).

NUM_REQUESTS = 1000 #600
# Number of requests to sample into the trace.

RATE_QPS = 3
# Arrival rate in requests/second. Record i arrives at time i / RATE_QPS.

SLO_CLASSES = {
    "strict":  {"ttft_ms": 600.0, "tpot_ms":  80.0, "weight": 0.30},
    "normal":  {"ttft_ms": 1000.0, "tpot_ms": 100.0, "weight": 0.50},
    "relaxed": {"ttft_ms": 1500.0, "tpot_ms": 150.0, "weight": 0.20},
}
_SLO_NAMES = list(SLO_CLASSES.keys())
_SLO_WEIGHTS = [v["weight"] for v in SLO_CLASSES.values()]

MIN_PROMPT_CHARS = 512
# Minimum prompt length in characters (filter out short prompts).

MAX_PROMPT_CHARS = 6000
# Maximum prompt length in characters (filter out very long prompts).

SEED = 42
# Random seed for reproducibility (shuffling, SLO sampling, token counts).

DATASET_DIR = "data/sharegpt52k"
# Local path to the ShareGPT52K dataset (downloaded once via huggingface_hub).

TOKENIZER_DIR = "/home/ubuntu/lqs/LLM_model"
# Local tokenizer used to convert dataset reference outputs into max_tokens.

REPO_ID = "RyokoAI/ShareGPT52K"
# Hugging Face repository ID (re-downloaded if Git LFS pointers detected).
# ==============================================================================

USER_ROLES = ("human", "user")
ASSISTANT_ROLES = ("gpt", "assistant")


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



def message_role(msg: dict) -> str:
    return msg.get("from", msg.get("role", "")).lower()


def message_content(msg: dict) -> str:
    return msg.get("value", msg.get("content", ""))


def find_reference_output(messages: list[dict], start_idx: int) -> str:
    for msg in messages[start_idx + 1:]:
        role = message_role(msg)
        if role in ASSISTANT_ROLES:
            return message_content(msg)
        if role in USER_ROLES:
            return ""
    return ""


def main():
    random.seed(SEED)

    _ensure_dataset()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

    # Load dataset
    candidates = []
    ds_dir = Path(DATASET_DIR)
    for fpath in sorted(ds_dir.glob("*.json")):
        with open(fpath) as f:
            convos = json.load(f)
        for convo in convos:
            messages = convo.get("conversations", convo.get("messages", []))
            for idx, msg in enumerate(messages):
                role = message_role(msg)
                prompt = message_content(msg)
                if role in USER_ROLES:
                    if MIN_PROMPT_CHARS <= len(prompt) <= MAX_PROMPT_CHARS:
                        output = find_reference_output(messages, idx)
                        output_tokens = len(
                            tokenizer.encode(output, add_special_tokens=False))
                        if output_tokens > 0:
                            candidates.append((prompt, output_tokens))
                    break

    print(f"[prepare_dataset] Found {len(candidates)} eligible prompt/output pairs")
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
        for i, (prompt, max_tokens) in enumerate(prompts):
            arrival_s = i / RATE_QPS
            slo_class = random.choices(_SLO_NAMES, weights=_SLO_WEIGHTS, k=1)[0]
            slo = SLO_CLASSES[slo_class]
            record = {
                "id": f"req_{i:06d}",
                "arrival_s": round(arrival_s, 6),
                "prompt": prompt,
                "max_tokens": max_tokens,
                "slo_class": slo_class,
                "ttft_ms": slo["ttft_ms"],
                "tpot_ms": slo["tpot_ms"],
                "w_n": 1.0,
            }
            f.write(json.dumps(record) + "\n")

    print(f"[prepare_dataset] Wrote {num} records to {OUTPUT}")


if __name__ == "__main__":
    main()
