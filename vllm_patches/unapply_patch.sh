#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${1:?Usage: unapply_patch.sh <vllm-dir>}"

echo "[unapply_patch] Removing energy_sched package from ${VLLM_DIR} ..."

# 1. Remove package directory
rm -rf "${VLLM_DIR}/vllm/energy_sched"

# 2. Remove sentinel-guarded hook from vllm/__init__.py
python3 -c "
import re, sys
path = '${VLLM_DIR}/vllm/__init__.py'
with open(path) as f:
    content = f.read()
pattern = r'\n# <<< ENERGY_SCHED_HOOK >>>.*$'
new_content = re.sub(pattern, '', content, flags=re.DOTALL)
if new_content != content:
    with open(path, 'w') as f:
        f.write(new_content)
    print('[unapply_patch] Hook removed from vllm/__init__.py')
else:
    print('[unapply_patch] No hook found — already clean.')
"

echo "[unapply_patch] Done."
