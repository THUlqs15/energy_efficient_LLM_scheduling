"""Collect per-run metrics from results.jsonl, power.csv, and optional iter log."""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional


def trapz(xs, ys):
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        total += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return total


def interpolate_power(ts, xs, ys):
    if not xs:
        return None
    if ts < xs[0] or ts > xs[-1]:
        return None
    if ts == xs[0]:
        return ys[0]
    if ts == xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= ts:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            ratio = (ts - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return None


def windowed_energy(xs, ys, start_ts, end_ts):
    if len(xs) < 2 or start_ts is None or end_ts is None or end_ts <= start_ts:
        return 0.0, 0.0
    start_p = interpolate_power(start_ts, xs, ys)
    end_p = interpolate_power(end_ts, xs, ys)
    if start_p is None or end_p is None:
        return 0.0, 0.0

    win_xs = [start_ts]
    win_ys = [start_p]
    for x, y in zip(xs, ys):
        if start_ts < x < end_ts:
            win_xs.append(x)
            win_ys.append(y)
    win_xs.append(end_ts)
    win_ys.append(end_p)

    energy = trapz(win_xs, win_ys)
    duration = end_ts - start_ts
    mean_power = energy / duration if duration > 0 else 0.0
    return energy, mean_power


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

    reqs = []
    with open(args.requests) as f:
        for line in f:
            reqs.append(json.loads(line))

    completed = [
        r for r in reqs
        if r.get("status") == 200 and r.get("error") is None
    ]
    num_completed = len(completed)
    num_failed = len(reqs) - num_completed

    valid_ttft = [r for r in completed if r.get("ttft_ms") is not None]
    valid_tpot = [r for r in completed if r.get("tpot_ms") is not None]

    ttfts = [r["ttft_ms"] for r in valid_ttft]
    tpots = [r["tpot_ms"] for r in valid_tpot]

    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0

    ttft_violations = [max(0.0, r["ttft_ms"] - r["ttft_slo_ms"]) for r in valid_ttft]
    tpot_violations = [max(0.0, r["tpot_ms"] - r["tpot_slo_ms"]) for r in valid_tpot]
    mean_ttft_viol = sum(ttft_violations) / len(ttft_violations) if ttft_violations else 0.0
    mean_tpot_viol = sum(tpot_violations) / len(tpot_violations) if tpot_violations else 0.0

    ttft_attain = sum(1 for v in ttft_violations if v == 0.0) / len(ttft_violations) if ttft_violations else 0.0
    tpot_attain = sum(1 for v in tpot_violations if v == 0.0) / len(tpot_violations) if tpot_violations else 0.0

    power_ts = []
    power_ws = []
    if os.path.exists(args.power):
        import csv
        with open(args.power) as f:
            reader = csv.DictReader(f)
            for row in reader:
                power_ts.append(float(row["timestamp_s"]))
                power_ws.append(float(row["power_w"]))

    first_send = min(
        (r.get("send_time") for r in reqs if r.get("send_time") is not None),
        default=None,
    )
    last_complete = max(
        (r.get("complete_time") for r in reqs if r.get("complete_time") is not None),
        default=None,
    )

    if power_ts:
        total_energy, mean_power = windowed_energy(
            power_ts,
            power_ws,
            first_send,
            last_complete,
        )
    else:
        total_energy = 0.0
        mean_power = 0.0

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
