"""Energy-aware frequency-first scheduler for vLLM.

Two layers:
  (a) Pure-Python core — FrequencyFirstSolver (unit-testable, no vLLM imports).
  (b) vLLM integration — EnergyScheduler subclass built via factory.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# === (a) Pure-Python core ===================================================

from .energy_model import (
    LatencyParams,
    PowerParams,
    per_request_time_ms,
    batch_overhead_ms,
    load_latency_params,
    load_power_params,
)
from .frequency_controller import get_controller


@dataclass
class EnergySchedConfig:
    beta: float = 1.0
    w_ttft: float = 1.0
    w_tpot: float = 1.0
    eta_ms: float = 1e9
    Lmax: int = 0
    default_w_n: float = 1.0
    default_ttft_ms: float = 4000.0
    default_tpot_ms: float = 200.0
    freq_candidates: Optional[List[int]] = None
    freq_stride: int = 1
    log_every_n: int = 50
    iter_log_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EnergySchedConfig":
        return cls(
            beta=float(os.environ.get("VLLM_ENERGY_BETA", "1.0")),
            w_ttft=float(os.environ.get("VLLM_ENERGY_W_TTFT", "1.0")),
            w_tpot=float(os.environ.get("VLLM_ENERGY_W_TPOT", "1.0")),
            eta_ms=float(os.environ.get("VLLM_ENERGY_ETA_MS", "1e9")),
            Lmax=int(os.environ.get("VLLM_ENERGY_LMAX", "0")),
            freq_stride=int(os.environ.get("VLLM_ENERGY_FREQ_STRIDE", "1")),
            iter_log_path=os.environ.get("VLLM_ENERGY_ITER_LOG"),
        )


@dataclass
class ReqView:
    handle: Any
    is_prefill: bool
    l_q: int
    l_kv: int
    wait_ms: float
    deadline_ms: float
    w_n: float
    kv_blocks_needed: int = 0


def instant_utility(r: ReqView, cfg: EnergySchedConfig) -> float:
    r_n = r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)
    slack_ms = r.deadline_ms - r.wait_ms
    return r_n * math.exp(-slack_ms / 1000.0)


def adjusted_utility(
    r: ReqView, cfg: EnergySchedConfig, f_mhu: float,
    latency: LatencyParams, power: PowerParams,
) -> Tuple[float, float]:
    t_q = per_request_time_ms(latency, f_mhu, r.is_prefill, r.l_q, r.l_kv)
    f_in = instant_utility(r, cfg)
    v = f_in - cfg.beta * power.power_watts(f_mhu) * t_q
    return v, t_q


def greedy_knapsack_2d(
    reqs: List[Any],
    values: List[float],
    times_ms: List[float],
    tokens: List[int],
    Lmax: int,
    eta_ms: float,
) -> List[int]:
    indices = list(range(len(values)))
    items = []
    for i in indices:
        if values[i] <= 0:
            continue
        denom = max(tokens[i] / Lmax, times_ms[i] / eta_ms)
        if denom <= 0:
            denom = 1e-12
        items.append((values[i] / denom, i))
    items.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    used_tokens = 0
    used_time = 0.0
    for _, i in items:
        if used_tokens + tokens[i] <= Lmax and used_time + times_ms[i] <= eta_ms:
            chosen.append(i)
            used_tokens += tokens[i]
            used_time += times_ms[i]
    return chosen


class FrequencyFirstSolver:
    def __init__(
        self,
        cfg: EnergySchedConfig,
        latency: LatencyParams,
        power: PowerParams,
        freq_candidates: List[int],
    ):
        self.cfg = cfg
        self.latency = latency
        self.power = power
        self.freq_candidates = freq_candidates

    def solve(self, reqs: List[ReqView], Lmax: int) -> Tuple[float, list, float]:
        # Separate running (decode) from waiting (prefill) requests.
        # Running requests MUST be served to avoid stalling in-flight generations.
        running_reqs = [(i, r) for i, r in enumerate(reqs) if not r.is_prefill]
        prefill_reqs = [(i, r) for i, r in enumerate(reqs) if r.is_prefill]

        best = (0.0, self.freq_candidates[-1] if self.freq_candidates else 1410, [], 0.0)
        has_prefill_any = len(prefill_reqs) > 0
        has_decode_any = len(running_reqs) > 0
        masks = []
        if has_prefill_any:
            masks.append("prefill_only")
        if has_decode_any:
            masks.append("decode_only")
        if has_prefill_any and has_decode_any:
            masks.append("mixed")

        stride = self.cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates

        for f in freqs:
            P_f = self.power.power_watts(f)
            v_t = [adjusted_utility(r, self.cfg, f, self.latency, self.power) for r in reqs]
            for M in masks:
                has_p = M in ("prefill_only", "mixed")
                has_d = M in ("decode_only", "mixed")
                T_ovh = batch_overhead_ms(self.latency, f, has_p, has_d)
                eta_left = self.cfg.eta_ms - T_ovh - self.latency.t_c
                if eta_left <= 0:
                    continue

                # Build candidate set for knapsack:
                # - Running (decode) requests are always mandatory
                # - Prefill requests are optional (energy-aware selection)
                mandatory_idxs = [i for i, r in running_reqs] if has_d else []
                optional_idxs = [i for i, r in prefill_reqs] if has_p else []

                if M == "prefill_only":
                    idxs = optional_idxs
                    mandatory_in_idxs = []
                elif M == "decode_only":
                    idxs = mandatory_idxs
                    mandatory_in_idxs = list(range(len(idxs)))
                else:  # mixed
                    idxs = mandatory_idxs + optional_idxs
                    mandatory_in_idxs = list(range(len(mandatory_idxs)))

                if not idxs:
                    continue

                # Separate values/times/tokens for mandatory vs optional
                values = []
                times = []
                tok = []
                mandatory_in_filtered = []
                opt_offset = len(mandatory_idxs)
                for j, i in enumerate(idxs):
                    r = reqs[i]
                    v, t = v_t[i]
                    if j < opt_offset:
                        # Mandatory item — always include
                        # Use large positive value to guarantee selection
                        values.append(abs(v) + 1000.0)
                        mandatory_in_filtered.append(j)
                    elif v <= 0:
                        continue  # skip optional prefill with negative utility
                    else:
                        values.append(v)
                    times.append(t)
                    tok.append(r.l_q)

                if not values:
                    continue

                # Run greedy knapsack
                local_map = list(range(len(values)))
                picked_local = greedy_knapsack_2d(
                    local_map, values, times, tok, Lmax, eta_left
                )

                # Check that all mandatory items were picked
                mandatory_set = set(mandatory_in_filtered)
                if mandatory_set and not mandatory_set.issubset(set(picked_local)):
                    continue

                picked = [idxs[j] for j in picked_local]

                if M == "mixed" and picked:
                    chosen_reqs = [reqs[i] for i in picked]
                    if not (any(r.is_prefill for r in chosen_reqs)
                            and any(not r.is_prefill for r in chosen_reqs)):
                        continue

                sum_v = sum(v_t[i][0] for i in picked)
                J = sum_v - self.cfg.beta * P_f * (T_ovh + self.latency.t_c)
                if J > best[0]:
                    et_pred = (
                        sum(v_t[i][1] for i in picked) + T_ovh + self.latency.t_c
                    )
                    best = (J, f, picked, et_pred)

        _, f_star, picked, et_pred = best
        return float(f_star), [reqs[i] for i in picked], et_pred


# === (b) vLLM integration ===================================================

def _open_iter_log(path: Optional[str]):
    if path is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "a")


def make_energy_scheduler_class():
    from vllm.v1.core.sched.scheduler import Scheduler

    class EnergyScheduler(Scheduler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cfg = EnergySchedConfig.from_env()
            self._latency = load_latency_params()
            self._power = load_power_params()
            self._freq_ctl = get_controller()
            cands = (
                self._cfg.freq_candidates
                or self._freq_ctl.supported_clocks()
                or [1410]
            )
            self._solver = FrequencyFirstSolver(
                self._cfg, self._latency, self._power, cands
            )
            if self._cfg.Lmax <= 0:
                self._cfg.Lmax = int(getattr(
                    self.scheduler_config, "max_num_batched_tokens",
                    getattr(self.scheduler_config, "max_model_len", 8192),
                ))
            self._iter_log = _open_iter_log(self._cfg.iter_log_path)
            self._prev_exit_t = None
            self._iter = 0
            self._prev_record = None

        def _build_request_views(self, now_ms: float) -> List[ReqView]:
            reqs: List[ReqView] = []
            block_size = getattr(self, "block_size", 16)
            for req in self.waiting:
                extra = getattr(req, "sampling_params", None)
                ea = getattr(extra, "extra_args", {}) if extra else {}
                if isinstance(ea, dict):
                    ttft = ea.get("ttft_ms", self._cfg.default_ttft_ms)
                    tpot = ea.get("tpot_ms", self._cfg.default_tpot_ms)
                    w_n = ea.get("w_n", self._cfg.default_w_n)
                else:
                    ttft = self._cfg.default_ttft_ms
                    tpot = self._cfg.default_tpot_ms
                    w_n = self._cfg.default_w_n
                arrival = getattr(req, "arrival_time", now_ms / 1000.0) * 1000.0
                wait_ms = now_ms - arrival
                l_q = getattr(req, "num_prompt_tokens", 0)
                l_kv = 0
                kv_blocks = (l_q + block_size - 1) // block_size
                reqs.append(ReqView(
                    handle=req, is_prefill=True, l_q=l_q, l_kv=l_kv,
                    wait_ms=wait_ms, deadline_ms=ttft, w_n=w_n,
                    kv_blocks_needed=kv_blocks,
                ))
            for req in self.running:
                extra = getattr(req, "sampling_params", None)
                ea = getattr(extra, "extra_args", {}) if extra else {}
                if isinstance(ea, dict):
                    ttft = ea.get("ttft_ms", self._cfg.default_ttft_ms)
                    tpot = ea.get("tpot_ms", self._cfg.default_tpot_ms)
                    w_n = ea.get("w_n", self._cfg.default_w_n)
                else:
                    ttft = self._cfg.default_ttft_ms
                    tpot = self._cfg.default_tpot_ms
                    w_n = self._cfg.default_w_n
                arrival = getattr(req, "arrival_time", now_ms / 1000.0) * 1000.0
                wait_ms = now_ms - arrival
                l_kv = getattr(req, "num_computed_tokens", 0)
                l_q = 1
                kv_blocks = (l_kv + block_size) // block_size
                reqs.append(ReqView(
                    handle=req, is_prefill=False, l_q=l_q, l_kv=l_kv,
                    wait_ms=wait_ms, deadline_ms=tpot, w_n=w_n,
                    kv_blocks_needed=kv_blocks,
                ))
            return reqs

        def _kv_evict(
            self, chosen: List[ReqView], f_mhu: float
        ) -> List[ReqView]:
            kv_mgr = getattr(self, "kv_cache_manager", None)
            if kv_mgr is None:
                return chosen
            block_pool = getattr(kv_mgr, "block_pool", None)
            if block_pool is None:
                return chosen
            free_fn = getattr(block_pool, "get_num_free_blocks", None)
            if free_fn is None:
                return chosen
            while chosen:
                total = sum(r.kv_blocks_needed for r in chosen)
                free = free_fn()
                if total <= free:
                    break
                v_t = []
                for r in chosen:
                    t_q = per_request_time_ms(
                        self._latency, f_mhu, r.is_prefill, r.l_q, r.l_kv
                    )
                    v = instant_utility(r, self._cfg) - self._cfg.beta * self._power.power_watts(f_mhu) * t_q
                    v_t.append((v, r))
                v_t.sort(key=lambda x: x[0])
                chosen = [r for _, r in v_t[1:]]
            return chosen

        def _materialise_batch(self, chosen: List[ReqView]):
            waiting_handles = {r.handle for r in chosen if r.is_prefill}
            running_handles = {r.handle for r in chosen if not r.is_prefill}
            saved_waiting = [
                r for r in self.waiting if r not in waiting_handles
            ]
            saved_running = [
                r for r in self.running if r not in running_handles
            ]
            # Remove unchosen requests from waiting and running
            self.waiting.remove_requests(saved_waiting)
            for r in saved_running:
                self.running.remove(r)
            try:
                out = super().schedule()
            finally:
                for r in saved_waiting:
                    self.waiting.add_request(r)
                self.running.extend(saved_running)
            return out

        def schedule(self):
            t_enter = time.monotonic()
            exec_ms = (
                (t_enter - self._prev_exit_t) * 1000.0
                if self._prev_exit_t is not None else None
            )
            now_ms = time.time() * 1000.0
            reqs = self._build_request_views(now_ms)
            t_solve0 = time.monotonic()
            f_star, chosen, et_pred = self._solver.solve(reqs, self._cfg.Lmax)
            solve_ms = (time.monotonic() - t_solve0) * 1000.0
            self._freq_ctl.set_frequency(int(f_star))
            # Fallback to default scheduling if solver returns empty batch
            if not chosen:
                out = super().schedule()
            else:
                chosen = self._kv_evict(chosen, f_star)
                out = self._materialise_batch(chosen)
            if self._iter_log is not None:
                if self._prev_record is not None and exec_ms is not None:
                    rec = self._prev_record
                    rec["exec_ms"] = exec_ms
                    self._iter_log.write(json.dumps(rec) + "\n")
                    self._iter_log.flush()
                self._prev_record = {
                    "iter": self._iter,
                    "solve_ms": solve_ms,
                    "batch_size": len(chosen),
                    "f_star": int(f_star),
                    "et_pred_ms": et_pred,
                }
                self._iter += 1
            self._prev_exit_t = time.monotonic()
            if self._iter_log is not None and self._iter % self._cfg.log_every_n == 0:
                exec_str = f"{exec_ms:.2f}" if exec_ms else "N/A"
                print(
                    f"[energy_sched] iter={self._iter} f*={int(f_star)} "
                    f"|B|={len(chosen)} solve_ms={solve_ms:.2f} "
                    f"exec_ms={exec_str}",
                    flush=True,
                )
            return out

    return EnergyScheduler
