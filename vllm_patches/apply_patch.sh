#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${1:?Usage: apply_patch.sh <vllm-dir>}"

echo "[apply_patch] Installing energy_sched package into ${VLLM_DIR} ..."

# 0. Clear Python bytecode cache so Python doesn't load stale .pyc files
rm -rf "${VLLM_DIR}/vllm/energy_sched/__pycache__"
find "${VLLM_DIR}/vllm/energy_sched" -name '*.pyc' -delete 2>/dev/null || true

# 1. Copy package files
mkdir -p "${VLLM_DIR}/vllm/energy_sched"
cp "$(dirname "$0")/__init__.py"       "${VLLM_DIR}/vllm/energy_sched/__init__.py"
cp "$(dirname "$0")/energy_model.py"    "${VLLM_DIR}/vllm/energy_sched/energy_model.py"
cp "$(dirname "$0")/energy_scheduler.py" "${VLLM_DIR}/vllm/energy_sched/energy_scheduler.py"
cp "$(dirname "$0")/frequency_controller.py" "${VLLM_DIR}/vllm/energy_sched/frequency_controller.py"

# 2. Append sentinel-guarded hook to vllm/__init__.py
MARKER="# <<< ENERGY_SCHED_HOOK >>>"
if grep -qF "$MARKER" "${VLLM_DIR}/vllm/__init__.py"; then
    echo "[apply_patch] Hook already present in vllm/__init__.py — skipping."
else
    cat >> "${VLLM_DIR}/vllm/__init__.py" <<'PYHOOK'

# <<< ENERGY_SCHED_HOOK >>>
import os as _ENERGY_os
if _ENERGY_os.environ.get("VLLM_ENERGY_SCHEDULER", "0") == "1":
    try:
        from vllm.energy_sched.energy_scheduler import make_energy_scheduler_class
        from vllm.v1.core.sched import scheduler as _ENERGY_mod
        _ENERGY_mod.Scheduler = make_energy_scheduler_class()
    except Exception as _e:
        import sys as _s
        print(f"[energy_sched] failed to install: {_e}", file=_s.stderr, flush=True)
PYHOOK
    echo "[apply_patch] Hook appended to vllm/__init__.py"
fi

echo "[apply_patch] Done."
