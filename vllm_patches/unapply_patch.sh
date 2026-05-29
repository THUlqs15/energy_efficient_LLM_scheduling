#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${1:?Usage: unapply_patch.sh <vllm-dir>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[unapply_patch] Reverting energy_sched from ${VLLM_DIR} ..."

# 1. Reverse the scheduler patch
PATCH_FILE="${PATCH_DIR}/scheduler_energy.patch"
if [[ -f "$PATCH_FILE" ]]; then
    if cd "${VLLM_DIR}" && ! git diff --quiet vllm/v1/core/sched/scheduler.py 2>/dev/null; then
        echo "[unapply_patch] Reversing scheduler_energy.patch ..."
        git apply -R "${PATCH_FILE}" || {
            echo "[unapply_patch] git apply -R failed; restoring from git ..."
            git checkout -- vllm/v1/core/sched/scheduler.py
        }
    else
        echo "[unapply_patch] scheduler.py is already clean — skipping."
    fi
fi

# 2. Remove energy_sched package
if [[ -d "${VLLM_DIR}/vllm/energy_sched" ]]; then
    rm -rf "${VLLM_DIR}/vllm/energy_sched" 2>/dev/null || {
        # Handle permission issues with __pycache__
        find "${VLLM_DIR}/vllm/energy_sched" -type f -delete 2>/dev/null || true
        find "${VLLM_DIR}/vllm/energy_sched" -type d -empty -delete 2>/dev/null || true
    }
    echo "[unapply_patch] Removed vllm/energy_sched/"
fi

# 3. Remove old monkey-patch hook from vllm/__init__.py if still present
MARKER="# <<< ENERGY_SCHED_HOOK >>>"
if grep -qF "$MARKER" "${VLLM_DIR}/vllm/__init__.py" 2>/dev/null; then
    sed -i "/${MARKER}/,\$d" "${VLLM_DIR}/vllm/__init__.py"
    echo "[unapply_patch] Removed old monkey-patch hook from vllm/__init__.py"
fi

echo "[unapply_patch] Done. vLLM is restored to its original state."
