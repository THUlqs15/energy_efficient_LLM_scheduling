"""pynvml wrapper for GPU SM frequency control."""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Optional

try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class FrequencyController:
    """Locks GPU SM clock via NVML."""

    def __init__(self, gpu_index: int = 0):
        if not _HAS_PYNVML:
            raise RuntimeError("pynvml is not installed")
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._gpu_idx = gpu_index
        self._last_f: Optional[int] = None
        self._clocks = self._query_supported_clocks()

    # -- public ----------------------------------------------------------------

    def supported_clocks(self) -> list[int]:
        return sorted(self._clocks)

    def set_frequency(self, f_mhz: int) -> bool:
        target = self._closest(f_mhz)
        if target == self._last_f:
            return True
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(self._handle, target, target)
            self._last_f = target
            return True
        except Exception:
            # Fall back to nvidia-smi with sudo
            try:
                result = subprocess.run(
                    ["sudo", "nvidia-smi", "-lgc", str(target), "-i", str(self._gpu_idx)],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self._last_f = target
                    return True
            except Exception:
                pass
            print(f"[freq_ctl] set_frequency({target}) failed — skipping freq lock")
            self._last_f = target
            return False

    def reset(self):
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
        except Exception:
            try:
                subprocess.run(
                    ["sudo", "nvidia-smi", "-rgc", "-i", str(self._gpu_idx)],
                    capture_output=True, text=True, timeout=5
                )
            except Exception:
                pass
        self._last_f = None

    # -- internal --------------------------------------------------------------

    def _closest(self, f_mhu: int) -> int:
        if not self._clocks:
            return f_mhu
        return min(self._clocks, key=lambda c: abs(c - f_mhu))

    @staticmethod
    def _query_supported_clocks() -> list[int]:
        try:
            return pynvml.nvmlDeviceGetSupportedGraphicsClocks(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )
        except Exception:
            pass
        # fallback: parse nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-supported-clocks=gr", "--format=csv,noheader"],
                text=True,
            )
            return [int(x.strip()) for x in out.splitlines() if x.strip()]
        except Exception:
            return []


@lru_cache(maxsize=1)
def get_controller() -> FrequencyController:
    idx = int(os.environ.get("VLLM_ENERGY_GPU_INDEX", "0"))
    return FrequencyController(idx)
