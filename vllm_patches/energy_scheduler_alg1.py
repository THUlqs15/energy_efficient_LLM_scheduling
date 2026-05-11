"""Energy-aware frequency-first scheduler for vLLM.

UPDATED VERSION — solver hot path optimized via:
  (1) Pulling frequency-independent quantities (instant utility f_in,
      latency-numerator t_num) out of the frequency loop — computed once
      per solve() call instead of F times.
  (2) NumPy vectorization of the inner per-request loop. For each candidate
      frequency we now do O(1) vector ops over N requests instead of N
      Python function calls. P_f is a scalar computed once per f, broadcast
      across the request vector — so the redundant power_watts(f) calls
      that were happening inside adjusted_utility for every (f, req) pair
      are gone for free.

Two layers (unchanged):
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

import numpy as np


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
    max_batch_size: int = 0   # 0 / <=0 → inherit vLLM scheduler_config.max_num_seqs
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
            max_batch_size=int(os.environ.get("VLLM_ENERGY_MAX_BATCH_SIZE", "0")),
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


# --- legacy scalar helpers (kept for backward compat / unit tests) ----------

def instant_utility(r: ReqView, cfg: EnergySchedConfig) -> float:
    r_n = r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)
    slack_ms = r.deadline_ms - r.wait_ms
    return r_n * min(math.exp(-slack_ms / 1000.0), 5.0)


def adjusted_utility(
    r: ReqView, cfg: EnergySchedConfig, f_mhu: float,
    latency: LatencyParams, power: PowerParams,
) -> Tuple[float, float]:
    t_q = per_request_time_ms(latency, f_mhu, r.is_prefill, r.l_q, r.l_kv)
    t_q_s = t_q / 1000.0  # ms → s for correct energy (Joules)
    f_in = instant_utility(r, cfg)
    v = f_in - cfg.beta * power.power_watts(f_mhu) * t_q_s
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


# --- numpy-vectorized greedy knapsack (used in hot path) --------------------

def _greedy_knapsack_2d_np(
    sub_v: np.ndarray,
    sub_t: np.ndarray,
    sub_tok: np.ndarray,
    Lmax: int,
    eta_ms: float,
    B_max: int,
) -> List[int]:
    """Same logic as greedy_knapsack_2d but on numpy arrays, with an extra
    request-count cap B_max (<=0 means no cap).

    Inputs are already pre-filtered to v > 0. Returns local indices into the
    sub_* arrays (caller is responsible for mapping back to global indices).
    """
    if sub_v.size == 0:
        return []
    denom = np.maximum(sub_tok / float(Lmax), sub_t / float(eta_ms))
    # protect against zero/negative denom
    denom = np.where(denom <= 0.0, 1e-12, denom)
    ratio = sub_v / denom
    # descending sort, stable to match Python's sorted(reverse=True) tie-break
    order = np.argsort(-ratio, kind="stable")

    # the greedy fill itself is sequential — unavoidable, but small
    chosen: List[int] = []
    used_tokens = 0
    used_time = 0.0
    tok_sorted = sub_tok[order]
    t_sorted = sub_t[order]
    cap = B_max if B_max > 0 else order.size  # <=0 ⇒ effectively unlimited
    for k in range(order.size):
        if len(chosen) >= cap:
            break  # batch-size cap hit — no point checking remaining items
        tk = int(tok_sorted[k])
        tt = float(t_sorted[k])
        if used_tokens + tk <= Lmax and used_time + tt <= eta_ms:
            chosen.append(int(order[k]))
            used_tokens += tk
            used_time += tt
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

    def solve(self, reqs: List[ReqView], Lmax: int, debug_iter: int = -1) -> Tuple[float, list, float]:
        default_f = self.freq_candidates[-1] if self.freq_candidates else 1410
        if not reqs:
            return float(default_f), [], 0.0

        N = len(reqs)
        cfg = self.cfg
        lat = self.latency
        beta = cfg.beta
        t_c = lat.t_c

        # === (1) Pre-compute frequency-INDEPENDENT per-request quantities ===
        # All as numpy arrays so the per-frequency body becomes vector ops.
        is_pf = np.fromiter((r.is_prefill for r in reqs), dtype=bool, count=N)
        l_q = np.fromiter((r.l_q for r in reqs), dtype=np.float64, count=N)
        l_kv = np.fromiter((r.l_kv for r in reqs), dtype=np.float64, count=N)
        w_n = np.fromiter((r.w_n for r in reqs), dtype=np.float64, count=N)
        deadline = np.fromiter((r.deadline_ms for r in reqs), dtype=np.float64, count=N)
        wait = np.fromiter((r.wait_ms for r in reqs), dtype=np.float64, count=N)
        tok_arr = np.fromiter((r.l_q for r in reqs), dtype=np.int64, count=N)

        # f_in = instant_utility, vectorized
        r_n = w_n * np.where(is_pf, cfg.w_ttft, cfg.w_tpot)
        slack_ms = deadline - wait
        f_in = r_n * np.minimum(np.exp(-slack_ms / 1000.0), 5.0)

        # t_num = numerator of per_request_time_ms, freq-independent
        # prefill: a_p*l_q^2 + b_p*l_q*l_kv + c_p*l_q
        # decode:  a_d*l_kv  + b_d
        t_num_pf = lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q
        t_num_dc = lat.a_d * l_kv + lat.b_d
        t_num = np.where(is_pf, t_num_pf, t_num_dc)

        # Mode masks
        has_prefill_any = bool(is_pf.any())
        has_decode_any = bool((~is_pf).any())
        masks: List[str] = []
        if has_prefill_any:
            masks.append("prefill_only")
        if has_decode_any:
            masks.append("decode_only")
        if has_prefill_any and has_decode_any:
            masks.append("mixed")

        # Frequency candidates (subsampled by stride)
        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates

        # Dynamic eta — tightest slack across all reqs, lower-bounded by cfg.eta_ms
        min_slack = float(slack_ms.min())
        effective_eta = max(cfg.eta_ms, min_slack)

        # Effective batch-size cap. cfg.max_batch_size <= 0 means "no cap".
        B_max = cfg.max_batch_size if cfg.max_batch_size > 0 else 0

        best_J = 0.0
        best_f = default_f
        best_picked: List[int] = []
        best_et_pred = 0.0

        # === (2) Frequency loop — each iteration is vectorized over N reqs ===
        for f in freqs:
            P_f = self.power.power_watts(f)        # scalar, computed once per f
            f_alpha = f ** lat.alpha               # scalar

            # t_q[i] = t_num[i] / f          if prefill
            #         t_num[i] / f^alpha     if decode
            denom_t = np.where(is_pf, float(f), f_alpha)
            t_q = t_num / denom_t                  # ms

            # v[i] = f_in[i] - beta * P_f * t_q[i] / 1000
            v = f_in - beta * P_f * (t_q / 1000.0)

            for M in masks:
                T_ovh = batch_overhead_ms(
                    lat, f, M != "decode_only", M != "prefill_only"
                )
                eta_left = effective_eta - T_ovh - t_c
                if eta_left <= 0:
                    continue

                # Boolean selector: (matches mask) AND (v > 0)
                if M == "prefill_only":
                    sel = is_pf & (v > 0.0)
                elif M == "decode_only":
                    sel = (~is_pf) & (v > 0.0)
                else:  # mixed
                    sel = v > 0.0

                local_idx = np.nonzero(sel)[0]
                if local_idx.size == 0:
                    continue

                sub_v = v[local_idx]
                sub_t = t_q[local_idx]
                sub_tok = tok_arr[local_idx]

                picked_in_sub = _greedy_knapsack_2d_np(
                    sub_v, sub_t, sub_tok, Lmax, eta_left, B_max
                )
                if not picked_in_sub:
                    continue

                # Map sub-indices back to global request indices
                picked_arr = local_idx[picked_in_sub]

                if M == "mixed":
                    picked_is_pf = is_pf[picked_arr]
                    if not (picked_is_pf.any() and (~picked_is_pf).any()):
                        continue

                sum_v = float(v[picked_arr].sum())
                J = sum_v - beta * P_f * (T_ovh + t_c) / 1000.0
                if J > best_J:
                    et_pred = float(t_q[picked_arr].sum() + T_ovh + t_c)
                    best_J = J
                    best_f = f
                    best_picked = picked_arr.tolist()
                    best_et_pred = et_pred

        # DEBUG: trace which mode wins and J values
        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = int(is_pf[best_picked].sum()) if best_picked else 0
            n_d_ch = len(best_picked) - n_p_ch
            print(
                f"[dbg] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"picked={len(best_picked)}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} eff_eta={min(effective_eta, cfg.eta_ms):.0f} "
                f"min_sl={min_slack:.0f} B_max={B_max if B_max > 0 else 'inf'}",
                file=sys.stderr, flush=True)

        return float(best_f), [reqs[i] for i in best_picked], best_et_pred


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
            if self._cfg.max_batch_size <= 0:
                self._cfg.max_batch_size = int(getattr(
                    self.scheduler_config, "max_num_seqs", 128
                ))
            self._iter_log = _open_iter_log(self._cfg.iter_log_path)
            self._prev_exit_t = None
            self._iter = 0
            self._prev_record = None
            self._last_exec = {}

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
                req_id = getattr(req, "request_id", id(req))
                last_exec_ms = self._last_exec.get(req_id)
                if last_exec_ms is not None:
                    wait_ms = now_ms - last_exec_ms
                else:
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
                    v = instant_utility(r, self._cfg) - self._cfg.beta * self._power.power_watts(f_mhu) * (t_q / 1000.0)
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
            f_star, chosen, et_pred = self._solver.solve(reqs, self._cfg.Lmax, self._iter)
            solve_ms = (time.monotonic() - t_solve0) * 1000.0
            self._freq_ctl.set_frequency(int(f_star))
            # Fallback to default scheduling if solver returns empty batch
            if not chosen:
                out = super().schedule()
            else:
                chosen = self._kv_evict(chosen, f_star)
                out = self._materialise_batch(chosen)
                for r in chosen:
                    req_id = getattr(r.handle, "request_id", id(r.handle))
                    self._last_exec[req_id] = now_ms
            if self._iter_log is not None:
                if self._prev_record is not None and exec_ms is not None:
                    rec = self._prev_record
                    rec["exec_ms"] = exec_ms
                    self._iter_log.write(json.dumps(rec) + "\n")
                    self._iter_log.flush()
                if chosen:
                    self._prev_record = {
                        "iter": self._iter,
                        "solve_ms": solve_ms,
                        "batch_size": len(chosen),
                        "f_star": int(f_star),
                        "et_pred_ms": et_pred,
                    }
                else:
                    self._prev_record = None
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
