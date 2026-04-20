"""Fitted Route-B+ latency model and cubic power model.

All coefficients are hard-coded defaults from the A800 profiling pipeline
(§4.1 / §4.2 of experiment_instruction.md).  Optional JSON overrides are
read from $VLLM_ENERGY_LATENCY_JSON / $VLLM_ENERGY_POWER_JSON.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Sequence, Tuple


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------

@dataclass
class LatencyParams:
    """9-parameter Route B+ batch-latency model."""
    a_p:   float = 4.791065541701025e-03
    b_p:   float = 0.0
    c_p:   float = 1.3650620923913607e+02
    w_pf:  float = 1.4999985410892014e+04
    w_dec: float = 1.4999998752209698e+04
    a_d:   float = 1.9294172833774345e-01
    b_d:   float = 5.0502340019606976e+01
    alpha: float = 0.9735669988793928
    t_c:   float = 4.652569884043852

    @classmethod
    def from_json(cls, path: str) -> "LatencyParams":
        with open(path) as f:
            data = json.load(f)
        params = data.get("params", data)
        defaults = {k: v for k, v in asdict(cls()).items()}
        for k in defaults:
            if k not in params:
                params[k] = defaults[k]
        return cls(**params)


# ---------------------------------------------------------------------------
# Power model
# ---------------------------------------------------------------------------

@dataclass
class PowerParams:
    """Cubic polynomial: P(f) = k3·f³ + k2·f² + k1·f + k0."""
    k3: float = 1.711824e-07
    k2: float = -3.252635e-04
    k1: float = 3.194042e-01
    k0: float = 3.789920e+01

    def power_watts(self, f_mhz: float) -> float:
        return (
            self.k3 * f_mhz ** 3
            + self.k2 * f_mhz ** 2
            + self.k1 * f_mhz
            + self.k0
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def per_request_time_ms(latency: LatencyParams, freq_mhz: float,
                        is_prefill: bool, l_q: int, l_kv: int) -> float:
    """Per-request t_q(n, f) only. Excludes w_pf / w_dec / t_c."""
    if is_prefill:
        return (latency.a_p * l_q * l_q
                + latency.b_p * l_q * l_kv
                + latency.c_p * l_q) / freq_mhz
    return (latency.a_d * l_kv + latency.b_d) / (freq_mhz ** latency.alpha)


def batch_overhead_ms(latency: LatencyParams, freq_mhz: float,
                      has_prefill: bool, has_decode: bool) -> float:
    """T_ovh(B, f) = w_pf·1{has_prefill}/f + w_dec·1{has_decode}/f^alpha."""
    ovh = 0.0
    if has_prefill:
        ovh += latency.w_pf / freq_mhz
    if has_decode:
        ovh += latency.w_dec / (freq_mhz ** latency.alpha)
    return ovh


def batch_time_ms(latency: LatencyParams, freq_mhz: float,
                  requests: Sequence[Tuple[bool, int, int]]) -> float:
    """Total ET_i(B, f) = sum t_q + T_ovh(B, f) + t_c."""
    if not requests:
        return 0.0
    has_prefill = any(r[0] for r in requests)
    has_decode = any(not r[0] for r in requests)
    total = sum(per_request_time_ms(latency, freq_mhz, *r) for r in requests)
    total += batch_overhead_ms(latency, freq_mhz, has_prefill, has_decode)
    total += latency.t_c
    return total


def load_latency_params() -> LatencyParams:
    path = os.environ.get("VLLM_ENERGY_LATENCY_JSON")
    if path and os.path.isfile(path):
        return LatencyParams.from_json(path)
    return LatencyParams()


def load_power_params() -> PowerParams:
    path = os.environ.get("VLLM_ENERGY_POWER_JSON")
    if path and os.path.isfile(path):
        return PowerParams.from_json(path)
    return PowerParams()
