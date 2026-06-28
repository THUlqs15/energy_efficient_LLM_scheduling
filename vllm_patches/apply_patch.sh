#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${1:?Usage: apply_patch.sh <vllm-dir>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[apply_patch] Installing energy_sched package into ${VLLM_DIR} ..."

# 0. Clear Python bytecode cache so Python doesn't load stale .pyc files
rm -rf "${VLLM_DIR}/vllm/energy_sched/__pycache__" 2>/dev/null || true
find "${VLLM_DIR}/vllm/energy_sched" -name '*.pyc' -delete 2>/dev/null || true

# 1. Copy energy_sched package files
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
SCHEDULER_PATH="vllm/v1/core/sched/scheduler.py"
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[apply_patch] ERROR: scheduler_energy.patch not found at ${PATCH_FILE}"
    exit 1
fi

if cd "${VLLM_DIR}" && git apply --check "${PATCH_FILE}" 2>/dev/null; then
    echo "[apply_patch] Applying scheduler_energy.patch ..."
    git apply "${PATCH_FILE}"
    echo "[apply_patch] Patch applied successfully."
elif cd "${VLLM_DIR}" && git apply -R --check "${PATCH_FILE}" 2>/dev/null; then
    echo "[apply_patch] scheduler_energy.patch already applied — skipping."
elif grep -q "_energy_enabled" "${VLLM_DIR}/${SCHEDULER_PATH}" 2>/dev/null; then
    echo "[apply_patch] Existing energy scheduler patch is stale — refreshing scheduler.py ..."
    backup_path="/tmp/scheduler.py.energy_stale.$(date +%s).bak"
    cp "${VLLM_DIR}/${SCHEDULER_PATH}" "$backup_path"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    mkdir -p "${tmpdir}/$(dirname "${SCHEDULER_PATH}")"
    git -C "${VLLM_DIR}" show "HEAD:${SCHEDULER_PATH}" \
        > "${tmpdir}/${SCHEDULER_PATH}"
    git -C "$tmpdir" apply "${PATCH_FILE}"
    cp "${tmpdir}/${SCHEDULER_PATH}" "${VLLM_DIR}/${SCHEDULER_PATH}"
    echo "[apply_patch] Refreshed scheduler.py (backup: ${backup_path})."
else
    echo "[apply_patch] ERROR: scheduler.py has unrelated local changes; refusing to overwrite."
    echo "[apply_patch] Revert or inspect ${VLLM_DIR}/${SCHEDULER_PATH}, then re-run."
    exit 1
fi

echo "[apply_patch] Done."
