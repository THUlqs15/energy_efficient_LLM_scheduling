"""Energy-aware scheduler for vLLM — ALTERNATIVE FORMULATION 1.

Implements the breakpoint-enumeration algorithm of Section 2 of
energy-efficient.tex (Alt 1: SOFT exp deadline penalty, breakpoint-only
τ search per the latest LaTeX revision — interior stationary-point search
via bisection/Newton has been DROPPED as a tractable approximation).

Algorithm:

    For each τ ∈ T = {s_n}:
      • Compute u_n(τ) for all n:
            u_n(τ) = r_n                          if τ ≤ s_n
                   = r_n · exp(-(τ - s_n))        if τ > s_n
      • Sort requests by u_n(τ)/ℓ_{i,n} descending.
      • Run incremental greedy:
            For each n in sorted order, with current state B_cur:
                If |B_cur| < B_max and Σℓ + ℓ_n ≤ L_max:
                    f*(B_cur ∪ {n}, τ) = min{ f ∈ F : ET(B,f) ≤ τ }
                    E(B_cur ∪ {n}, τ) = P(f*) · τ
                    Δ_n = u_n(τ) − β · [E(B_cur∪{n}, τ) − E(B_cur, τ)]
                    If Δ_n > 0: accept n.
      • OPT(τ) = Σ_{n∈B*} u_n(τ) − β · E(B*, τ).
    Pick τ* = argmax_τ OPT(τ); return B* and f*(B*, τ*).

s_n = deadline_n − T_{i,n}  (slack convention, positive = on time).
NOTE: The text "denote s_n = T - deadline" in the LaTeX is a sign typo;
the algorithm box's "s_n = deadline - T" is the consistent convention,
required for both the indicator I{ET ≤ s_n} (Alt 2) and the soft form
exp(-[τ - s_n]_+) (Alt 1) to be physically meaningful.

vs. Section-1 (frequency-first) this module REPLACES:
  • The eta_ms (η) hyper-parameter is GONE — τ is enumerated from data.
  • For backward compat, env var VLLM_ENERGY_ETA_MS is still accepted,
    just ignored by the solver, so main.sh works unchanged.

Compute-cost optimisations vs. naive Alt-1 (without altering the SOLUTION
LOGIC — i.e., greedy still examines requests in u_n/ℓ-density order and
admits each iff Δ_n > 0 ∧ token/batch-size feasible):

  (1) De-duplicate τ candidates and drop τ ≤ t_c (overhead alone exceeds
      budget — feasibility region is empty).
  (2) Pre-vectorise freq-INDEPENDENT per-request quantities (r_n, s_n,
      ℓ_n, the workload contributions wp_contrib, wd_contrib).
  (3) Vectorised u_n(τ) per τ (one np.exp call after [τ-s_n]_+ clip).
  (4) Vectorised ratio sort key u_n/ℓ_n via np.argsort.
  (5) Maintain (W_p, W_d, I_p, I_d, used_tok) INCREMENTALLY across the
      greedy — no re-summation per addition attempt.
  (6) Pre-cache P(f) for every f in F (avoid k0+k1·f+... recomputation).
  (7) f*(B_new, τ) lookup uses a "try previous freq first" fast path:
      if ET(B_new, f_prev) ≤ τ then f*_new = f_prev; otherwise scan only
      f_prev+1 .. f_max (since adding requests can only increase f*, never
      decrease). Saves |F|-1 array ops per addition in the common case.
  (8) Early `break` when |B| reaches B_max (no further additions possible
      regardless of remaining order).
  (9) Early `continue` when u_n ≤ 0 (Δ_n ≤ 0 since ΔE ≥ 0).
"""
from __future__ import annotations

import json
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
    eta_ms: float = 1e9          # accepted from env for backward compat; UNUSED in Alt-1
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
    """r_n = w_n · (w_TTFT for prefill, w_TPOT for decode)."""
    return r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)


# === The Alt-1 solver =======================================================

class Alt1Solver:
    """Alt-1 breakpoint-enumeration solver (soft exp deadline penalty).

    Per τ in the breakpoint set T = {s_n}:
      • Vectorise u_n(τ) and density sort key.
      • Run incremental greedy (sequential — preserved per algorithm spec).
      • Track OPT(τ) = Σ u_n − β · E(B*, τ).
    Choose τ* maximising OPT, return the matching B* and f*(B*, τ*).
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
        # Sort freqs ascending so that f*(·) lookup is "smallest feasible"
        self.freq_candidates = sorted(freq_candidates)

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

        # s_n: slack = deadline_n - T_{i,n}; positive = on time
        s_n = deadline - wait

        # Per-request workload contributions to (W_p(B), W_d(B)):
        #   prefill: a_p·l_q² + b_p·l_q·l_kv + c_p·l_q   (added when n is prefill)
        #   decode : a_d·l_kv  + b_d·l_q                 (added when n is decode)
        wp_contrib = np.where(
            is_pf, lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q, 0.0
        )
        wd_contrib = np.where(
            is_pf, 0.0, lat.a_d * l_kv + lat.b_d * l_q
        )

        # ---- (2) τ candidates: dedup, drop τ ≤ t_c (overhead-infeasible) --
        tau_pool = s_n[s_n > t_c]
        if tau_pool.size == 0:
            return float(default_f), [], 0.0
        tau_candidates = np.unique(tau_pool)  # ascending

        # ---- (3) Frequency grid (sorted ascending, with stride) -----------
        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates
        f_arr = np.asarray(freqs, dtype=np.float64)
        f_alpha = f_arr ** lat.alpha
        F_size = f_arr.size
        # Pre-cache P(f) for all freqs
        P_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )

        # ---- (4) Effective Bmax cap ---------------------------------------
        B_eff = int(Bmax) if Bmax > 0 else N

        # ---- (5) Outer enumeration over τ ---------------------------------
        best_J = 0.0          # empty batch ⇒ OPT = 0; only positive wins
        best_f = float(default_f)
        best_picked: List[int] = []
        best_et = 0.0

        # Allocate reusable scratch buffers (avoid per-iter allocation)
        ET_buf = np.empty(F_size, dtype=np.float64)

        for tau in tau_candidates:
            tau = float(tau)

            # (a) u_n(τ) — vectorised. [τ - s_n]_+ then exp(-·/1000).
            #     IMPORTANT: τ and s_n are in ms; the soft-penalty time-scale
            #     is seconds (consistent with Section 1's `instant_utility`,
            #     which uses exp(-slack_ms/1000)). Without this /1000, the
            #     exp degenerates to a hard step function — any ms-level
            #     overshoot drives the reward to ~0.
            #     Non-penalised (s_n ≥ τ): exp(0) = 1 → u_n = r_n.
            #     Penalised      (s_n < τ): exp(-(τ - s_n)/1000) ∈ (0,1).
            exponent = tau - s_n
            np.maximum(exponent, 0.0, out=exponent)         # in-place clip [·]_+
            u_n = r_n_vec * np.exp(-exponent / 1000.0)

            # (b) Sort by u_n/ℓ_{i,n} descending (stable for tie determinism)
            density = u_n / np.maximum(tok_arr.astype(np.float64), 1.0)
            order = np.argsort(-density, kind="stable")

            # (c) Incremental greedy. State maintained incrementally:
            W_p = 0.0
            W_d = 0.0
            has_pf = False
            has_dc = False
            used_tok = 0
            picked_local: List[int] = []
            sum_u = 0.0
            E_prev = 0.0     # P(f*) · τ in JOULES (= W·s, so divided by 1000)
            f_prev_idx = -1  # < 0 ⇒ no previous freq; full freq scan needed

            for idx in order:
                # Cheap constraint checks first
                if len(picked_local) >= B_eff:
                    break  # batch full; remaining order can't add anything
                tok_n = int(tok_arr[idx])
                if used_tok + tok_n > Lmax:
                    continue  # this n exceeds tokens; smaller n may still fit
                u_idx = float(u_n[idx])
                if u_idx <= 0.0:
                    continue  # u_n=0 ⇒ Δ_n ≤ 0 since ΔE ≥ 0; reject

                # Tentative new state after adding n
                idx_pf = bool(is_pf[idx])
                if idx_pf:
                    new_W_p = W_p + float(wp_contrib[idx])
                    new_W_d = W_d
                    new_has_pf = True
                    new_has_dc = has_dc
                else:
                    new_W_p = W_p
                    new_W_d = W_d + float(wd_contrib[idx])
                    new_has_pf = has_pf
                    new_has_dc = True

                num_p = new_W_p + (lat.w_pf if new_has_pf else 0.0)
                num_d = new_W_d + (lat.w_dec if new_has_dc else 0.0)

                # f*(B_new, τ) lookup with "try previous freq first" fast path.
                # Adding requests monotonically increases ET at every f, so the
                # smallest feasible f never decreases as the batch grows.
                f_new_idx = -1
                if f_prev_idx >= 0:
                    ET_at_prev = (
                        num_p / f_arr[f_prev_idx]
                        + num_d / f_alpha[f_prev_idx]
                        + t_c
                    )
                    if ET_at_prev <= tau:
                        f_new_idx = f_prev_idx

                if f_new_idx < 0:
                    start = f_prev_idx + 1 if f_prev_idx >= 0 else 0
                    if start >= F_size:
                        continue  # already pinned at f_max, can't go higher
                    # Compute ET on the [start, F_size) slice into ET_buf
                    sub = ET_buf[start:F_size]
                    np.divide(num_p, f_arr[start:F_size], out=sub)
                    sub += num_d / f_alpha[start:F_size]
                    sub += t_c
                    feasible = sub <= tau
                    if not feasible.any():
                        continue
                    f_new_idx = start + int(np.argmax(feasible))

                P_new = float(P_arr[f_new_idx])
                # E in J (= W·s) → divide ms by 1000
                E_new = P_new * tau / 1000.0

                # Δ_n = u_n − β · (E_new − E_prev)
                delta = u_idx - beta * (E_new - E_prev)
                if delta <= 0.0:
                    continue

                # Accept
                picked_local.append(int(idx))
                W_p, W_d = new_W_p, new_W_d
                has_pf, has_dc = new_has_pf, new_has_dc
                used_tok += tok_n
                sum_u += u_idx
                E_prev = E_new
                f_prev_idx = f_new_idx

            if not picked_local:
                continue

            # OPT(τ) per LaTeX: Σ u_n(τ) − β · E(B*, τ)  ← E uses τ, not actual ET
            OPT = sum_u - beta * E_prev
            if OPT <= best_J:
                continue

            # Record best. Also compute the ACTUAL ET at f_prev (≤ τ) for monitoring.
            num_p = W_p + (lat.w_pf if has_pf else 0.0)
            num_d = W_d + (lat.w_dec if has_dc else 0.0)
            ET_actual = (
                num_p / f_arr[f_prev_idx]
                + num_d / f_alpha[f_prev_idx]
                + t_c
            )

            best_J = OPT
            best_f = float(f_arr[f_prev_idx])
            best_picked = picked_local
            best_et = float(ET_actual)

        # Debug
        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in best_picked if reqs[i].is_prefill)
            n_d_ch = len(best_picked) - n_p_ch
            print(
                f"[dbg-alt1] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"picked={len(best_picked)}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} |T|={tau_candidates.size} "
                f"OPT*={best_J:.3f} B_max={B_eff}",
                file=sys.stderr, flush=True)

        return float(best_f), [reqs[i] for i in best_picked], best_et


# Alias kept so external code that imports `FrequencyFirstSolver` still works
FrequencyFirstSolver = Alt1Solver


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
            self._solver = Alt1Solver(
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
                # Evict the request with the lowest baseline reward (proxy).
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
                    f"[energy_sched-alt1] iter={self._iter} f*={int(f_star)} "
                    f"|B|={len(chosen)} solve_ms={solve_ms:.2f} "
                    f"exec_ms={exec_str}",
                    flush=True,
                )
            return out

    return EnergyScheduler
