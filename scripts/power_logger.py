"""Sample GPU power / clock / utilisation via pynvml and write CSV."""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time

try:
    import pynvml
except ImportError:
    print("[power_logger] pynvml not installed", file=sys.stderr)
    sys.exit(1)

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.1)
    p.add_argument("--output", default="power.csv")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "power_w", "sm_clock_mhz", "utilization_pct"])
        t0 = time.time()
        while not _stop:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                continue
            writer.writerow([f"{time.time() - t0:.4f}", f"{power:.2f}", clock, util])
            f.flush()
            time.sleep(args.interval)

    print(f"[power_logger] Wrote {args.output}")


if __name__ == "__main__":
    main()
