"""Compare default vs custom summaries, print table, write comparison.csv."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

METRIC_ORDER = [
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_ttft_violation_ms",
    "mean_tpot_violation_ms",
    "ttft_slo_attainment",
    "tpot_slo_attainment",
    "mean_power_w",
    "total_energy_j",
    "mean_solve_exec_ratio",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--default", default="results/summary_default.json")
    p.add_argument("--custom", default="results/summary_custom.json")
    p.add_argument("--output", default="comparison.csv")
    args = p.parse_args()

    summaries = {}
    for label, path in [("default", args.default), ("custom", args.custom)]:
        if os.path.exists(path):
            with open(path) as f:
                summaries[label] = json.load(f)
        else:
            print(f"[compare] WARNING: {path} not found", file=sys.stderr)
            summaries[label] = {}

    # Print side-by-side table
    header = f"{'metric':<30} {'default':>18} {'custom':>18}"
    print(header)
    print("-" * len(header))
    rows = []
    for m in METRIC_ORDER:
        dv = summaries.get("default", {}).get(m, "N/A")
        cv = summaries.get("custom", {}).get(m, "N/A")
        print(f"{m:<30} {dv:>18} {cv:>18}")
        rows.append((m, dv, cv))

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "default", "custom"])
        for m, dv, cv in rows:
            writer.writerow([m, dv, cv])

    print(f"\n[compare] Wrote {args.output}")


if __name__ == "__main__":
    main()
