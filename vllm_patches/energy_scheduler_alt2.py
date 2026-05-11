"""Energy-aware scheduler for vLLM — ALTERNATIVE FORMULATION 2.

Implements the threshold-enumeration algorithm of Section 3 (eqs 37-52) of
LLM_scheduling.pdf:

    For each (mode M, frequency f, threshold τ ∈ T = {s_n}):
        Solve sub-knapsack
            B(M,f,τ) = argmax_{B ⊆ N(M) ∩ N(τ)} Σ (r_n − β·P(f)·t_{n,f})
            s.t.   Σ ℓ_n ≤ Lmax,   |B| ≤ Bmax,
                   Σ t_{n,f} ≤ τ − t_c − o(M,f).
        Evaluate true J(M,f,τ) = Σ r_n − β·P(f)·ET_actual
            (indicator I{ET ≤ s_n} is automatically 1 for every n in B,
             because s_n ≥ τ ≥ ET).
    Pick (M*, f*, τ*) = argmax J;  return B* = B(M*, f*, τ*).

s_n  = deadline_n − T_{i,n}   (slack: positive = on-time, negative = overdue).
N(τ) = {n : s_n ≥ τ}.

Compared to the Section-1 (frequency-first) formulation this module REPLACES,
the eta_ms (η) hyper-parameter is GONE: τ is enumerated from data instead.
For backward compatibility the env var VLLM_ENERGY_ETA_MS is still accepted —
it is simply ignored by the solver (so main.sh works unchanged).

Compute-cost optimisations vs. a naive Alt-2 implementation:
  • Dedup τ candidates and drop τ ≤ 0 (overdue → infeasible eta_left).
  • Pull freq-independent quantities (r_n, s_n, t-numerator) out of the
    frequency loop.
  • NumPy-vectorise everything per-request that the loop body touches.
  • τ-INDEPENDENT pre-sort per (M, f) by v_n / max(ℓ_n/Lmax, 1/Bmax),
    reused across all τ (the time constraint is enforced inside greedy fill).
  • Early break when |B| hits Bmax.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


# === Pure-Python core =======================================================

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
    eta_ms: float = 1e9          # accepted from env for backward compat; UNUSED in Alt-2
    Lmax: int = 0
    max_batch_size: int = 0      # 0 → inherit from vLLM scheduler_config.max_num_seqs
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
            eta_ms=float(os.environ.get("VLLM_ENERGY_ETA_MS", "1e9")),  # ignored
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


# --- helper: baseline reward r_n (eq 2) ------------------------------------

def baseline_reward(r: ReqView, cfg: EnergySchedConfig) -> float:
    """r_n = w_n · (w_TTFT for prefill, w_TPOT for decode). No exp decay."""
    return r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)


# === The Alt-2 solver =======================================================

class Alt2Solver:
    """Alt-2 threshold-enumeration solver.

    For each (M, f, τ) it produces an approximate B(M,f,τ) via a 3-constraint
    greedy heuristic (tokens, batch-size, time). The τ-independent pre-sort
    inside makes the per-(M,f) cost O(|T| · N) Python work (very tight loop),
    matching the original frequency-first solver's order of magnitude.
    """

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

    def solve(
        self,
        reqs: List[ReqView],
        Lmax: int,
        Bmax: int,
        debug_iter: int = -1,
    ) -> Tuple[float, list, float]:
        default_f = self.freq_candidates[-1] if self.freq_candidates else 1410
        if not reqs:
            return float(default_f), [], 0.0

        N = len(reqs)
        cfg = self.cfg
        lat = self.latency
        beta = cfg.beta
        t_c = lat.t_c

        # ---- (1) Vectorise freq-INDEPENDENT per-request quantities --------
        is_pf = np.fromiter((r.is_prefill for r in reqs), dtype=bool, count=N)
        l_q = np.fromiter((r.l_q for r in reqs), dtype=np.float64, count=N)
        l_kv = np.fromiter((r.l_kv for r in reqs), dtype=np.float64, count=N)
        w_n = np.fromiter((r.w_n for r in reqs), dtype=np.float64, count=N)
        deadline = np.fromiter((r.deadline_ms for r in reqs), dtype=np.float64, count=N)
        wait = np.fromiter((r.wait_ms for r in reqs), dtype=np.float64, count=N)
        tok_arr = np.fromiter((r.l_q for r in reqs), dtype=np.int64, count=N)

        # r_n (eq 2): baseline reward
        r_n_vec = w_n * np.where(is_pf, cfg.w_ttft, cfg.w_tpot)

        # s_n: slack (= deadline_n − T_{i,n}); positive = on time
        s_n = deadline - wait

        # t_num: numerator of t_{n,f} (freq-independent)
        # prefill : a_p·l_q² + b_p·l_q·l_kv + c_p·l_q
        # decode  : a_d·l_kv  + b_d
        t_num_pf = lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q
        t_num_dc = lat.a_d * l_kv + lat.b_d
        t_num = np.where(is_pf, t_num_pf, t_num_dc)

        # ---- (2) Mode candidates (M ∈ {prefill_only, decode_only, mixed}) -
        has_pf = bool(is_pf.any())
        has_dc = bool((~is_pf).any())
        masks: List[str] = []
        if has_pf:
            masks.append("prefill_only")
        if has_dc:
            masks.append("decode_only")
        if has_pf and has_dc:
            masks.append("mixed")

        # ---- (3) Frequency candidates (subsampled by stride) --------------
        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates

        # ---- (4) τ candidates: dedup, drop ≤ 0 (always infeasible) --------
        # Eta_left = τ − t_c − o(M,f) > 0  ⇒  τ > t_c. We can pre-prune τ ≤ t_c too.
        tau_pool = s_n[s_n > t_c]
        if tau_pool.size == 0:
            return float(default_f), [], 0.0
        tau_candidates = np.unique(tau_pool)  # ascending

        # ---- (5) Effective Bmax cap ---------------------------------------
        B_eff = int(Bmax) if Bmax > 0 else N

        # ---- (6) τ-independent denom for pre-sort (token + batch slot only)
        # The time-budget term (which IS τ-dependent) is enforced inside the
        # greedy fill, not in the sort key. This costs a little optimality
        # vs. resorting per τ but is dramatically faster.
        denom_persist = np.maximum(
            tok_arr.astype(np.float64) / float(Lmax),
            1.0 / float(B_eff),
        )
        denom_persist = np.where(denom_persist <= 0.0, 1e-12, denom_persist)

        # ---- (7) Outer enumeration ----------------------------------------
        best_J = 0.0          # empty batch ⇒ J = 0; only positive-J wins
        best_f = default_f
        best_picked: List[int] = []
        best_et = 0.0

        for f in freqs:
            P_f = self.power.power_watts(f)
            f_alpha = f ** lat.alpha

            # t_{n,f} for all n  (ms)
            denom_t = np.where(is_pf, float(f), f_alpha)
            t_q = t_num / denom_t

            # v_n = r_n − β·P(f)·t_{n,f}/1000   (ms → s for J in coherent units)
            v_n = r_n_vec - beta * P_f * (t_q / 1000.0)

            for M in masks:
                I_p = 1 if M != "decode_only" else 0
                I_d = 1 if M != "prefill_only" else 0
                o_M_f = (lat.w_pf * I_p) / f + (lat.w_dec * I_d) / f_alpha

                if M == "prefill_only":
                    mode_sel = is_pf
                elif M == "decode_only":
                    mode_sel = ~is_pf
                else:
                    mode_sel = np.ones(N, dtype=bool)

                # Pre-filter: must match mode AND have v_n > 0 (negatives only hurt)
                eligible_mf = mode_sel & (v_n > 0.0)
                if not eligible_mf.any():
                    continue

                # τ-independent sort, masked to (M, v>0) eligibles only.
                # `order_mf` lists indices in descending priority, already
                # restricted to mode-eligible items.
                ratio = v_n / denom_persist
                # Tie-break: stable sort keeps original insertion order for ties.
                global_order = np.argsort(-ratio, kind="stable")
                order_mf = global_order[eligible_mf[global_order]]
                if order_mf.size == 0:
                    continue

                # Pre-extract sorted arrays (length K)
                K = order_mf.size
                t_mf = t_q[order_mf]                              # float64
                tok_mf = tok_arr[order_mf].astype(np.float64)     # cast for cumsum
                s_mf = s_n[order_mf]
                r_mf = r_n_vec[order_mf]
                ispf_mf = is_pf[order_mf]

                # ---- (8) Vectorised τ enumeration via prefix-cumsum -------
                # For each τ ∈ tau_candidates, build the eligibility mask
                # mask_mat[i, k] = (s_mf[k] ≥ tau_candidates[i]).  Then
                # cumulative sums (along k-axis) give the running totals
                # of an "include all eligible up to position k" greedy. We
                # use np.searchsorted to find the largest prefix length that
                # still satisfies (Lmax, B_eff, eta_left). This is a slightly
                # weaker heuristic than the per-element-skip variant — for
                # very heterogeneous l_q (mixed prefill batches with one
                # huge req) it may pick fewer reqs — but it lets the entire
                # τ loop run in numpy instead of Python, ~30× speedup.
                T_size = tau_candidates.size

                mask_mat = s_mf[None, :] >= tau_candidates[:, None]  # (T, K) bool

                cum_n = np.cumsum(mask_mat, axis=1, dtype=np.int64)         # |B| count
                cum_tok = np.cumsum(mask_mat * tok_mf[None, :], axis=1)
                cum_t = np.cumsum(mask_mat * t_mf[None, :], axis=1)
                cum_r = np.cumsum(mask_mat * r_mf[None, :], axis=1)

                if M == "mixed":
                    cum_pf = np.cumsum(mask_mat * ispf_mf[None, :].astype(np.int64),
                                       axis=1, dtype=np.int64)
                    cum_dc = np.cumsum(mask_mat * (~ispf_mf)[None, :].astype(np.int64),
                                       axis=1, dtype=np.int64)

                eta_left_vec = tau_candidates - t_c - o_M_f                # (T,)

                # Violations along each row: first True column = k_max
                viol = (
                    (cum_n > B_eff)
                    | (cum_tok > Lmax)
                    | (cum_t > eta_left_vec[:, None])
                )
                any_viol = viol.any(axis=1)
                first_viol = np.argmax(viol.astype(np.int8), axis=1)
                k_max = np.where(any_viol, first_viol, K)
                k_max[eta_left_vec <= 0.0] = 0    # τ infeasible by overhead alone

                # Compute prefix sums at index (k_max - 1), guarded for k_max=0
                idx_safe = np.maximum(k_max - 1, 0)
                rows = np.arange(T_size)
                sum_r_tau = cum_r[rows, idx_safe]
                sum_t_tau = cum_t[rows, idx_safe]
                if M == "mixed":
                    sum_pf_tau = cum_pf[rows, idx_safe]
                    sum_dc_tau = cum_dc[rows, idx_safe]
                zero = (k_max == 0)
                sum_r_tau = np.where(zero, 0.0, sum_r_tau)
                sum_t_tau = np.where(zero, 0.0, sum_t_tau)

                # True J per τ:  Σr − β·P(f)·ET_actual / 1000
                ET_tau = sum_t_tau + o_M_f + t_c
                J_tau = sum_r_tau - beta * P_f * ET_tau / 1000.0

                # Empty-batch fallback gives J = 0 → mask out negatives
                J_tau = np.where(zero, 0.0, J_tau)

                # Enforce mixed-mode constraint (both prefill AND decode present)
                if M == "mixed":
                    valid_mix = (sum_pf_tau > 0) & (sum_dc_tau > 0)
                    J_tau = np.where(valid_mix, J_tau, -np.inf)

                if J_tau.size == 0:
                    continue
                best_tau_idx = int(np.argmax(J_tau))
                J_here = float(J_tau[best_tau_idx])
                if J_here <= best_J:
                    continue

                # Recover the chosen indices for this winning τ
                kmax_win = int(k_max[best_tau_idx])
                if kmax_win == 0:
                    continue
                local_positions = np.nonzero(mask_mat[best_tau_idx, :kmax_win])[0]
                if local_positions.size == 0:
                    continue
                if M == "mixed":
                    ip = ispf_mf[local_positions]
                    if not (ip.any() and (~ip).any()):
                        continue

                best_J = J_here
                best_f = f
                best_et = float(ET_tau[best_tau_idx])
                best_picked = order_mf[local_positions].tolist()

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in best_picked if reqs[i].is_prefill)
            n_d_ch = len(best_picked) - n_p_ch
            print(
                f"[dbg-alt2] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"picked={len(best_picked)}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} |T|={tau_candidates.size} "
                f"J*={best_J:.3f} B_max={B_eff}",
                file=sys.stderr, flush=True)

        return float(best_f), [reqs[i] for i in best_picked], best_et


# Alias kept so external code that imports `FrequencyFirstSolver` still works
FrequencyFirstSolver = Alt2Solver


# === vLLM integration ======================================================

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
            self._solver = Alt2Solver(
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
                # Evict the request with the lowest baseline reward (no exp
                # decay in Alt-2, so just baseline_reward suffices as a proxy).
                v_t = []
                for r in chosen:
                    t_q = per_request_time_ms(
                        self._latency, f_mhu, r.is_prefill, r.l_q, r.l_kv
                    )
                    v = baseline_reward(r, self._cfg) - self._cfg.beta * self._power.power_watts(f_mhu) * (t_q / 1000.0)
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
            f_star, chosen, et_pred = self._solver.solve(
                reqs, self._cfg.Lmax, self._cfg.max_batch_size, self._iter
            )
            solve_ms = (time.monotonic() - t_solve0) * 1000.0
            self._freq_ctl.set_frequency(int(f_star))
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
                    f"[energy_sched-alt2] iter={self._iter} f*={int(f_star)} "
                    f"|B|={len(chosen)} solve_ms={solve_ms:.2f} "
                    f"exec_ms={exec_str}",
                    flush=True,
                )
            return out

    return EnergyScheduler
