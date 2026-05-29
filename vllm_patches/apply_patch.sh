#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${1:?Usage: apply_patch.sh <vllm-dir>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[apply_patch] Installing energy_sched package into ${VLLM_DIR} ..."

# 0. Clear Python bytecode cache so Python doesn't load stale .pyc files
rm -rf "${VLLM_DIR}/vllm/energy_sched/__pycache__" 2>/dev/null || true
find "${VLLM_DIR}/vllm/energy_sched" -name '*.pyc' -delete 2>/dev/null || true

# 1. Copy energy_sched package files (solver replaces energy_scheduler)
mkdir -p "${VLLM_DIR}/vllm/energy_sched"
cp "${PATCH_DIR}/__init__.py"             "${VLLM_DIR}/vllm/energy_sched/__init__.py"
cp "${PATCH_DIR}/energy_model.py"          "${VLLM_DIR}/vllm/energy_sched/energy_model.py"
cp "${PATCH_DIR}/solver.py"                "${VLLM_DIR}/vllm/energy_sched/solver.py"
cp "${PATCH_DIR}/frequency_controller.py"  "${VLLM_DIR}/vllm/energy_sched/frequency_controller.py"

# 2. Remove old monkey-patch hook from vllm/__init__.py if present
MARKER="# <<< ENERGY_SCHED_HOOK >>>"
if grep -qF "$MARKER" "${VLLM_DIR}/vllm/__init__.py"; then
    # Delete from the marker line to end of file (the hook was appended)
    sed -i "/${MARKER}/,\$d" "${VLLM_DIR}/vllm/__init__.py"
    echo "[apply_patch] Removed old monkey-patch hook from vllm/__init__.py"
fi

# 3. Apply scheduler.py patch (embedded energy branch)
PATCH_FILE="${PATCH_DIR}/scheduler_energy.patch"
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[apply_patch] ERROR: scheduler_energy.patch not found at ${PATCH_FILE}"
    exit 1
fi

# Check if patch is already applied
if cd "${VLLM_DIR}" && git diff --quiet vllm/v1/core/sched/scheduler.py 2>/dev/null; then
    echo "[apply_patch] Applying scheduler_energy.patch ..."
    git apply "${PATCH_FILE}"
    echo "[apply_patch] Patch applied successfully."
else
    echo "[apply_patch] scheduler.py already modified — skipping patch (re-run unapply_patch.sh first to re-apply)."
fi

echo "[apply_patch] Done."
