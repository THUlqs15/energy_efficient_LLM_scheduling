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
        self._mem_clocks = self._query_supported_memory_clocks()
        self._lock_memory_clock()

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
        try:
            pynvml.nvmlDeviceResetMemoryLockedClocks(self._handle)
        except Exception:
            try:
                subprocess.run(
                    ["sudo", "nvidia-smi", "-rmc", "-i", str(self._gpu_idx)],
                    capture_output=True, text=True, timeout=5
                )
            except Exception:
                pass
        self._last_f = None
        self._last_mem = None

    # -- internal --------------------------------------------------------------

    def _closest(self, f_mhu: int) -> int:
        if not self._clocks:
            return f_mhu
        return min(self._clocks, key=lambda c: abs(c - f_mhu))

    @staticmethod
    def _query_supported_clocks() -> list[int]:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # nvmlDeviceGetSupportedGraphicsClocks requires a memory clock argument;
            # passing 0 returns all supported graphics clocks.
            return pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, 0)
        except Exception:
            pass
        # fallback: parse nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-supported-clocks=gr", "--format=csv,noheader"],
                text=True,
            )
            # Output may include " MHz" suffix — strip it before int()
            clocks = []
            for line in out.splitlines():
                line = line.strip().replace(" MHz", "")
                if line:
                    clocks.append(int(line))
            return clocks
        except Exception:
            return []

    @staticmethod
    def _query_supported_memory_clocks() -> list[int]:
        """Returns supported memory clocks (A800 typically only reports [1593])."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        except Exception:
            return []

    def _lock_memory_clock(self):
        """Attempt to lock memory clock to 1593 MHz (profiling baseline).
        A800-SXM4 does not support memory clock locking — this is non-fatal.
        The memory clock remains at its default dynamic behaviour."""
        if not self._mem_clocks:
            return
        target = self._mem_clocks[0]
        try:
            pynvml.nvmlDeviceSetMemoryLockedClocks(self._handle, target, target)
            self._last_mem = target
        except Exception:
            try:
                subprocess.run(
                    ["sudo", "nvidia-smi", "-lmc", f"{target},{target}",
                     "-i", str(self._gpu_idx)],
                    capture_output=True, text=True, timeout=5,
                )
            except Exception:
                pass
            self._last_mem = None  # unsupported — expected on A800


@lru_cache(maxsize=1)
def get_controller() -> FrequencyController:
    idx = int(os.environ.get("VLLM_ENERGY_GPU_INDEX", "0"))
    return FrequencyController(idx)
