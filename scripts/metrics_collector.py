"""Collect per-run metrics from results.jsonl, power.csv, and optional iter log."""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional


def trapz(xs, ys):
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        total += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return total


def solve_exec_ratio(iter_log_path: Optional[str]) -> float:
    if not iter_log_path or not os.path.exists(iter_log_path):
        return 0.0
    ratios = []
    with open(iter_log_path) as f:
        for line in f:
            rec = json.loads(line)
            solve = rec.get("solve_ms")
            exec_ = rec.get("exec_ms")
            if solve is None or exec_ is None or exec_ <= 0.0:
                continue
            ratios.append(solve / exec_)
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", required=True)
    p.add_argument("--power", default="power.csv")
    p.add_argument("--iter-log", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--output", default="summary.json")
    args = p.parse_args()

    # Load results
    reqs = []
    with open(args.requests) as f:
        for line in f:
            reqs.append(json.loads(line))

    completed = [r for r in reqs if r.get("status") == 200 and r.get("error") is None]
    num_completed = len(completed)
    num_failed = len(reqs) - num_completed

    ttfts = [r["ttft_ms"] for r in completed if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in completed if r.get("tpot_ms") is not None]

    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    ttft_violations = [
        max(0.0, r["ttft_ms"] - r["ttft_slo_ms"])
        for r in completed if r.get("ttft_ms") is not None
    ]
    tpot_violations = [
        max(0.0, r["tpot_ms"] - r["tpot_slo_ms"])
        for r in completed if r.get("tpot_ms") is not None
    ]
    mean_ttft_viol = sum(ttft_violations) / len(ttft_violations) if ttft_violations else 0.0
    mean_tpot_viol = sum(tpot_violations) / len(tpot_violations) if tpot_violations else 0.0

    ttft_attain = sum(1 for v in ttft_violations if v == 0.0) / len(ttft_violations) if ttft_violations else 0.0
    tpot_attain = sum(1 for v in tpot_violations if v == 0.0) / len(tpot_violations) if tpot_violations else 0.0

    # Power integration
    power_ts = []
    power_ws = []
    if os.path.exists(args.power):
        import csv
        with open(args.power) as f:
            reader = csv.DictReader(f)
            for row in reader:
                power_ts.append(float(row["timestamp_s"]))
                power_ws.append(float(row["power_w"]))

    if power_ts:
        first_send = min(r.get("send_time", 0) for r in reqs) if reqs else 0
        last_complete = max(
            (r.get("complete_time", 0) or 0) for r in reqs
        ) if reqs else 0
        t0_abs = power_ts[0]
        # Convert to absolute: power_ts are relative to their own start
        # The power logger runs alongside the workload, so align by window
        energy = trapz(power_ts, power_ws)
        mean_power = energy / (power_ts[-1] - power_ts[0]) if power_ts[-1] > power_ts[0] else 0.0
        total_energy = energy
    else:
        mean_power = 0.0
        total_energy = 0.0

    ratio = solve_exec_ratio(args.iter_log)

    summary = {
        "label": args.label,
        "num_completed": num_completed,
        "num_failed": num_failed,
        "mean_ttft_ms": round(mean_ttft, 2),
        "mean_tpot_ms": round(mean_tpot, 2),
        "mean_ttft_violation_ms": round(mean_ttft_viol, 2),
        "mean_tpot_violation_ms": round(mean_tpot_viol, 2),
        "ttft_slo_attainment": round(ttft_attain, 4),
        "tpot_slo_attainment": round(tpot_attain, 4),
        "mean_power_w": round(mean_power, 2),
        "total_energy_j": round(total_energy, 2),
        "mean_solve_exec_ratio": round(ratio, 6),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[metrics_collector] {args.label}: {json.dumps(summary, indent=2)}")
    print(f"[metrics_collector] Wrote {args.output}")


if __name__ == "__main__":
    main()
