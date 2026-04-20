"""Async replay of trace.jsonl against a vLLM OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


@dataclass
class ResultRecord:
    id: str
    prompt: str
    max_tokens: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    w_n: float
    arrival_s: float
    send_time: float
    complete_time: Optional[float] = None
    status: Optional[int] = None
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    num_output_tokens: Optional[int] = None
    error: Optional[str] = None


async def send_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    record: dict,
    sem: asyncio.Semaphore,
) -> ResultRecord:
    async with sem:
        rr = ResultRecord(
            id=record["id"],
            prompt=record["prompt"],
            max_tokens=record["max_tokens"],
            ttft_slo_ms=record["ttft_ms"],
            tpot_slo_ms=record["tpot_ms"],
            w_n=record.get("w_n", 1.0),
            arrival_s=record["arrival_s"],
            send_time=time.monotonic(),
        )
        payload = {
            "model": model,
            "prompt": record["prompt"],
            "max_tokens": record["max_tokens"],
            "temperature": 0,
            "stream": True,
            "extra_body": {
                "extra_args": {
                    "ttft_ms": record["ttft_ms"],
                    "tpot_ms": record["tpot_ms"],
                    "w_n": record.get("w_n", 1.0),
                }
            },
        }
        try:
            first_chunk_time = None
            prev_chunk_time = None
            inter_chunk_gaps = []
            token_count = 0
            async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                rr.status = resp.status
                if resp.status != 200:
                    rr.error = await resp.text()
                    rr.complete_time = time.monotonic()
                    return rr
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = ""
                    choices = chunk.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                    now = time.monotonic()
                    if first_chunk_time is None:
                        first_chunk_time = now
                    if prev_chunk_time is not None:
                        inter_chunk_gaps.append((now - prev_chunk_time) * 1000.0)
                    prev_chunk_time = now
                    if text:
                        token_count += 1
            rr.complete_time = time.monotonic()
            if first_chunk_time is not None:
                rr.ttft_ms = (first_chunk_time - rr.send_time) * 1000.0
            rr.tpot_ms = (
                sum(inter_chunk_gaps) / len(inter_chunk_gaps)
                if inter_chunk_gaps else None
            )
            rr.num_output_tokens = token_count if token_count > 0 else None
        except Exception as e:
            rr.complete_time = time.monotonic()
            rr.error = str(e)
        return rr


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="trace.jsonl")
    p.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    p.add_argument("--model", default="default")
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--max-concurrency", type=int, default=256)
    args = p.parse_args()

    records = []
    with open(args.trace) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[workload_sender] Loaded {len(records)} records from {args.trace}")

    sem = asyncio.Semaphore(args.max_concurrency)
    t_start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = []
        idx = 0
        results = []
        for rec in records:
            idx += 1
            wait_until = rec["arrival_s"] - (time.monotonic() - t_start)
            if wait_until > 0:
                await asyncio.sleep(wait_until)
            task = asyncio.ensure_future(send_one(session, args.endpoint, args.model, rec, sem))
            tasks.append(task)
            if idx % 100 == 0:
                print(f"[workload_sender] Dispatched {idx}/{len(records)}", flush=True)
        results = await asyncio.gather(*tasks)

    # Write results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for rr in results:
            f.write(json.dumps({
                "id": rr.id,
                "prompt": rr.prompt,
                "max_tokens": rr.max_tokens,
                "ttft_slo_ms": rr.ttft_slo_ms,
                "tpot_slo_ms": rr.tpot_slo_ms,
                "w_n": rr.w_n,
                "arrival_s": rr.arrival_s,
                "send_time": rr.send_time,
                "complete_time": rr.complete_time,
                "status": rr.status,
                "ttft_ms": rr.ttft_ms,
                "tpot_ms": rr.tpot_ms,
                "num_output_tokens": rr.num_output_tokens,
                "error": rr.error,
            }) + "\n")

    completed = sum(1 for r in results if r.status == 200 and r.error is None)
    print(f"[workload_sender] Done: {completed}/{len(results)} completed, output → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
