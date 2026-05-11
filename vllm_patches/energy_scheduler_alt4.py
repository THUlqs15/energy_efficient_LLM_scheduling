"""Energy-aware scheduler for vLLM — ALTERNATIVE FORMULATION 4 (φ floor-clip).

Implements **candidate 4** of the user's φ generalisation,

    f_{i,n} = r_n · φ([T_{i,n} + ET(B,f) − deadline_n]_+),
    φ(x)    = max(1 − μ·x, φ_min),

i.e. a piecewise-linear soft deadline penalty with a floor.  Compared with
Alt-1 (alt3.py) which uses exp(-x/1000), and Alt-2 (alt2.py) which uses the
hard indicator I{x ≤ 0}, the floor-clipped linear φ has TWO kinks per
request, which makes the global τ-search exact at a finite candidate set:

    T = {s_n}_n  ∪  {s_n + r₀}_n          with   r₀ = (1 − φ_min) / μ
    |T| ≤ 2N.

Algorithm (mirrors Alt-2's (M, f, τ) triple-enumeration framework):

    For each (M, f, τ) ∈ M × F × T  (3 · |F| · 2N combinations):
      1. Compute u_n(τ) = r_n · max(1 − μ[τ − s_n]_+, φ_min)  for all n.
      2. Solve 2-D knapsack
            B(M,f,τ) = argmax_{B ⊆ N(M)} Σ (u_n − β·P(f)·t_{n,f})
            s.t.   Σ ℓ_n   ≤ Lmax,
                   |B|     ≤ Bmax,
                   Σ t_{n,f} ≤ τ − t_c − o(M,f).
      3. Score the true objective
            J(M,f,τ) = Σ u_n(τ) − β · P(f) · ET_actual(B,f).
    Pick (M*, f*, τ*) = argmax J;  return B* = B(M*, f*, τ*).

WHY THE BREAKPOINTS ARE SUFFICIENT
----------------------------------
For any FIXED B, J(τ; B) = Σu_n(τ) − β·P(f)·ET(B,f).  Only the first sum
depends on τ, and it is the sum of N piecewise-linear functions of τ with
KINKS only at {s_n} ∪ {s_n + r₀}.  Within each piece u_n′(τ) ∈ {0, −μ·r_n},
so ∂J/∂τ ≤ 0 for fixed B: J is non-increasing in τ, so the optimum on each
piece sits at its LEFT endpoint, which is one of the 2N breakpoints (or the
overhead floor τ = t_c + o, which we filter as infeasible separately).

UNIT CONVENTION  ⚠ DEPARTS FROM alt2/alt3 ⚠
--------------------------------------------
ALL time-related computations are performed in **SECONDS** (not ms):
  • wait_s, deadline_s, s_n, t_q_s, t_c_s, o_M_f_s, eta_left, τ, ET_actual.
  • φ slope μ has units of 1/seconds, so μ·x is dimensionless.
  • Energy = β·P(f)·ET_s in JOULES directly — no /1000 fudge anywhere.
The boundary with energy_model (which returns ms) is a single /1000 at the
point where t_num is divided by f^something to produce per-request time.

HARDCODED PARAMETERS OF φ (so main.sh need not change)
------------------------------------------------------
  • Alt4Solver.PHI_MU   : slope of φ in 1/s.  Bigger ⇒ steeper penalty.
  • Alt4Solver.PHI_MIN  : utility floor in (0, 1].  Bigger ⇒ even very-late
                          requests retain meaningful gradient information.

Compute-cost optimisations:
  (1) Dedup τ candidates and drop τ ≤ t_c (overhead alone exceeds budget).
  (2) Pull freq-INDEPENDENT per-request quantities (r_n, s_n, ℓ_n, t_num)
      out of the frequency loop.
  (3) Pre-compute u_n(τ) for ALL τ at once — shape (T, N), one np.maximum
      call.  φ’s floor-clip is just a max with the constant PHI_MIN.
  (4) For each (M, f), vectorise the entire τ enumeration via per-row
      argsort + cumsum (T × N matrix), so the inner loop is ONE numpy
      call instead of |T| Python iterations.
  (5) τ-dependent eligibility (mode AND v_n > 0) is handled by stuffing
      −∞ into the sort key, which buries inelig items at the end of each
      row's sorted order; their per-request totals are then masked to 0.
  (6) Mixed-mode constraint enforced via cum_pf, cum_dc cumsums.
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
    eta_ms: float = 1e9          # accepted from env for backward compat; UNUSED in Alt-4
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
    """r_n = w_n · (w_TTFT for prefill, w_TPOT for decode).  No exp / φ decay."""
    return r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)


# === The Alt-4 solver =======================================================

class Alt4Solver:
    """Alt-4 (M, f, τ)-enumeration with floor-clipped piecewise-linear penalty.

    All time arithmetic in **SECONDS**.  The energy_model returns ms; we divide
    by 1000 once at the boundary (per-request t_q, batch overhead w_pf/w_dec,
    batch constant t_c).
    """

    # === HARDCODED φ PARAMETERS (so main.sh need not change) ===
    # φ(x) = max(1 − μ·x, φ_min)         with x = [τ − s_n]_+ in SECONDS.
    # Tuning intent:
    #   • A 100-ms overshoot on a decode request drops u to 0.9 (still useful).
    #   • A 1-s overshoot on a prefill request drops u to ~0.1 (near the floor).
    #   • Linear decay region length r₀ = (1 − φ_min) / μ ≈ 0.9 s.
    PHI_MU: float = 1.0       # slope of φ, in 1/s.  Bigger ⇒ steeper penalty.
    PHI_MIN: float = 0.1      # utility floor in (0, 1].

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

        # ---- Time constants in SECONDS -----------------------------------
        t_c_s = lat.t_c / 1000.0   # batch constant overhead (s)

        # ---- (1) Vectorise freq-INDEPENDENT per-request quantities --------
        is_pf = np.fromiter((r.is_prefill for r in reqs), dtype=bool, count=N)
        l_q = np.fromiter((r.l_q for r in reqs), dtype=np.float64, count=N)
        l_kv = np.fromiter((r.l_kv for r in reqs), dtype=np.float64, count=N)
        w_n = np.fromiter((r.w_n for r in reqs), dtype=np.float64, count=N)
        deadline_s = np.fromiter(
            (r.deadline_ms / 1000.0 for r in reqs), dtype=np.float64, count=N
        )
        wait_s = np.fromiter(
            (r.wait_ms / 1000.0 for r in reqs), dtype=np.float64, count=N
        )
        tok_arr = np.fromiter((r.l_q for r in reqs), dtype=np.int64, count=N)

        # r_n: baseline reward
        r_n_vec = w_n * np.where(is_pf, cfg.w_ttft, cfg.w_tpot)

        # s_n: slack in SECONDS  (= deadline_s − wait_s); positive = on time
        s_n = deadline_s - wait_s

        # t_num: numerator of t_{n,f} (freq-independent, ms scale)
        # prefill : a_p·l_q² + b_p·l_q·l_kv + c_p·l_q
        # decode  : a_d·l_kv  + b_d
        t_num_pf = lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q
        t_num_dc = lat.a_d * l_kv + lat.b_d
        t_num = np.where(is_pf, t_num_pf, t_num_dc)
        # (We keep t_num at the ms scale and divide by 1000 with f below to
        # produce t_q_s in seconds.)

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

        # ---- (4) τ candidate set (in SECONDS, |T| ≤ 2N + 1) --------------
        # T = {s_n} ∪ {s_n + r₀}, drop τ ≤ t_c_s (overhead alone exceeds budget).
        # FALLBACK: if every natural breakpoint sits below t_c_s (i.e. ALL
        # requests are deeply overdue and already in φ's floor region), the
        # filtered pool is empty.  In that "all-φ_min" regime u_n(τ) = r_n·φ_min
        # is constant in τ, so any single feasible τ probes J correctly.  We add
        # one generous fallback τ_fb = t_c_s + 1.0 s, which gives ample
        # time-budget for any reasonable batch at any f (typical batch ET is
        # tens of ms).  Without this fallback the solver would unconditionally
        # return the empty batch under heavy backlog — pathological behaviour
        # that would let the queue grow without bound.
        mu = float(self.PHI_MU)
        phi_min = float(self.PHI_MIN)
        r0 = (1.0 - phi_min) / mu       # length of linear-decay region (s)
        tau_pool = np.concatenate([s_n, s_n + r0])
        tau_pool = tau_pool[tau_pool > t_c_s]
        if tau_pool.size == 0:
            tau_pool = np.array([t_c_s + 1.0])  # all-floor regime fallback
        tau_candidates = np.unique(tau_pool)   # ascending, deduped
        T_size = tau_candidates.size

        # ---- (5) Effective Bmax cap ---------------------------------------
        B_eff = int(Bmax) if Bmax > 0 else N

        # ---- (5b) float32 working copies for the (T, N) hot path ---------
        # Algorithm logic is unchanged; only the dtype is downgraded for the
        # dominant per-(M, f, τ) matrix arithmetic (overshoot, phi, u, v,
        # ratio, gather, cumsum).  f64 originals are retained for boundary
        # filtering / final J computation that benefit from full precision
        # (tau_pool > t_c_s, eta_left_vec, ET_tau_s, J_tau).
        F32 = np.float32
        r_n_vec_f32 = r_n_vec.astype(F32)
        s_n_f32 = s_n.astype(F32)
        t_num_f32 = t_num.astype(F32)
        tok_arr_f32 = tok_arr.astype(F32)
        tau_candidates_f32 = tau_candidates.astype(F32)
        mu_f32 = F32(self.PHI_MU)
        phi_min_f32 = F32(self.PHI_MIN)
        one_f32 = F32(1.0)
        zero_f32 = F32(0.0)
        inv_1000_f32 = F32(1.0 / 1000.0)
        beta_f32 = F32(beta)
        neg_inf_f32 = F32(-np.inf)

        # ---- (6) Pre-compute u_n(τ) for ALL τ at once --------------------
        # overshoot[t, n] = [τ_t − s_n]_+   (shape (T, N), in seconds, f32)
        overshoot = tau_candidates_f32[:, None] - s_n_f32[None, :]    # (T, N) f32
        np.maximum(overshoot, zero_f32, out=overshoot)
        # φ[t, n] = max(1 − μ · overshoot, φ_min)
        phi_mat = np.maximum(one_f32 - mu_f32 * overshoot, phi_min_f32)  # (T, N) f32
        u_mat = r_n_vec_f32[None, :] * phi_mat                        # (T, N) f32

        # ---- (7) τ-INDEPENDENT denom for sort heuristic -------------------
        # We use max(ℓ_n / Lmax, 1 / B_eff) — same as Alt-2.  The TIME budget
        # term (which IS τ-dependent) is enforced in the cumsum violation
        # check, not folded into the sort key.
        denom_persist = np.maximum(
            tok_arr.astype(np.float64) / float(Lmax),
            1.0 / float(B_eff),
        )
        denom_persist = np.where(denom_persist <= 0.0, 1e-12, denom_persist)
        denom_persist_f32 = denom_persist.astype(F32)

        # ---- (8) Outer enumeration ----------------------------------------
        best_J = 0.0          # empty batch ⇒ J = 0; only positive-J wins
        best_f = float(default_f)
        best_picked: List[int] = []
        best_et_s = 0.0

        n_arange = np.arange(N)
        rows = np.arange(T_size)

        for f in freqs:
            P_f = self.power.power_watts(f)
            f_alpha = f ** lat.alpha

            # Per-request t_{n,f} in SECONDS (f32)
            denom_t = np.where(is_pf, float(f), f_alpha).astype(F32)
            t_q_s = (t_num_f32 / denom_t) * inv_1000_f32         # (N,) f32

            # Per-request "energy cost" in JOULES (W · s) (f32)
            energy_cost = (beta_f32 * F32(P_f)) * t_q_s          # (N,) f32

            # v_n(τ, f) = u_n(τ) − energy_cost(n, f)             # (T, N) f32
            v_mat = u_mat - energy_cost[None, :]

            for M in masks:
                I_p = 1 if M != "decode_only" else 0
                I_d = 1 if M != "prefill_only" else 0
                # o(M, f) in SECONDS
                o_M_f_s = (
                    (lat.w_pf * I_p) / f + (lat.w_dec * I_d) / f_alpha
                ) / 1000.0

                if M == "prefill_only":
                    mode_sel = is_pf
                elif M == "decode_only":
                    mode_sel = ~is_pf
                else:
                    mode_sel = np.ones(N, dtype=bool)

                # Eligibility (T, N): mode AND profitable for this τ
                elig_mat = mode_sel[None, :] & (v_mat > 0.0)     # (T, N)
                if not elig_mat.any():
                    continue

                # Sort key per row: v / denom desc, with −∞ for non-elig.
                # This buries non-elig items at the END of each row's order,
                # so that prefix admission [0, k) only ever picks elig items
                # (after we mask their per-request quantities to 0 below).
                ratio_mat = np.where(
                    elig_mat, v_mat / denom_persist_f32[None, :], neg_inf_f32
                )
                order_mat = np.argsort(-ratio_mat, axis=1, kind="stable")  # (T, N) int64

                # Per-row gathered arrays.  For 1-D source arrays, fancy
                # indexing source[order_mat] returns shape (T, N) directly.
                gather_v = np.take_along_axis(v_mat, order_mat, axis=1)   # f32
                gather_u = np.take_along_axis(u_mat, order_mat, axis=1)   # f32
                gather_t = t_q_s[order_mat]                       # (T, N) f32
                gather_tok = tok_arr_f32[order_mat]               # (T, N) f32
                gather_pf = is_pf[order_mat]                      # (T, N) bool
                gather_elig = np.take_along_axis(elig_mat, order_mat, axis=1)

                # Mask non-elig contributions to 0 so they're inert in cumsum.
                # (np.where doesn't take an `out=` kwarg, so use *= mask.)
                elig_f = gather_elig.astype(F32)
                gather_v = gather_v * elig_f
                gather_u = gather_u * elig_f
                gather_t = gather_t * elig_f
                gather_tok = gather_tok * elig_f

                # Per-row cumulative sums (T, N)
                cum_n = np.cumsum(gather_elig.astype(np.int64), axis=1)
                cum_tok = np.cumsum(gather_tok, axis=1)
                cum_t = np.cumsum(gather_t, axis=1)
                cum_v = np.cumsum(gather_v, axis=1)
                cum_u = np.cumsum(gather_u, axis=1)

                # Time-budget per τ in SECONDS: τ − t_c_s − o(M, f)
                eta_left_vec = tau_candidates - t_c_s - o_M_f_s   # (T,)

                # Violation: any one of the 3 budgets crossed.
                viol = (
                    (cum_n > B_eff)
                    | (cum_tok > Lmax)
                    | (cum_t > eta_left_vec[:, None])
                )
                # Cap: don't grow past the elig count of the row.  cum_n
                # plateaus at elig_count, so we OR with "out-of-elig" mask.
                elig_count_per_row = cum_n[:, -1]                 # (T,)
                out_of_range = (
                    n_arange[None, :] >= elig_count_per_row[:, None]
                )
                viol_with_cap = viol | out_of_range

                any_viol = viol_with_cap.any(axis=1)
                first_viol = np.argmax(viol_with_cap.astype(np.int8), axis=1)
                k_max = np.where(any_viol, first_viol, N)
                # τ infeasible by overhead alone ⇒ k_max = 0
                k_max = np.where(eta_left_vec > 0.0, k_max, 0)
                # And cap at elig_count_per_row defensively
                k_max = np.minimum(k_max, elig_count_per_row)

                # Compute prefix sums at index (k_max − 1), guarded for k_max=0
                idx_safe = np.maximum(k_max - 1, 0)
                sum_u_tau = cum_u[rows, idx_safe]
                sum_t_tau = cum_t[rows, idx_safe]
                zero = (k_max == 0)
                sum_u_tau = np.where(zero, 0.0, sum_u_tau)
                sum_t_tau = np.where(zero, 0.0, sum_t_tau)

                # True J per τ in JOULES-aligned reward units
                ET_tau_s = sum_t_tau + o_M_f_s + t_c_s
                J_tau = sum_u_tau - beta * P_f * ET_tau_s
                J_tau = np.where(zero, 0.0, J_tau)

                # Mixed-mode constraint
                if M == "mixed":
                    cum_pf = np.cumsum(
                        (gather_pf & gather_elig).astype(np.int64), axis=1
                    )
                    cum_dc = np.cumsum(
                        ((~gather_pf) & gather_elig).astype(np.int64), axis=1
                    )
                    sum_pf_tau = cum_pf[rows, idx_safe]
                    sum_dc_tau = cum_dc[rows, idx_safe]
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
                # Eligibility mask on the first kmax_win positions of the
                # sorted row.  Even though non-elig items are buried later,
                # we still filter defensively in case kmax_win was clipped.
                elig_win = gather_elig[best_tau_idx, :kmax_win]
                local_positions = np.nonzero(elig_win)[0]
                if local_positions.size == 0:
                    continue
                picked_idx = order_mat[best_tau_idx, local_positions].tolist()
                if M == "mixed":
                    pf_picked = is_pf[picked_idx]
                    if not (pf_picked.any() and (~pf_picked).any()):
                        continue

                best_J = J_here
                best_f = float(f)
                best_picked = picked_idx
                best_et_s = float(ET_tau_s[best_tau_idx])

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in best_picked if reqs[i].is_prefill)
            n_d_ch = len(best_picked) - n_p_ch
            print(
                f"[dbg-alt4] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"picked={len(best_picked)}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} |T|={tau_candidates.size} "
                f"J*={best_J:.3f} B_max={B_eff} "
                f"μ={self.PHI_MU} φ_min={self.PHI_MIN}",
                file=sys.stderr, flush=True)

        # Convert ET back to ms for compatibility with the iter_log schema
        return float(best_f), [reqs[i] for i in best_picked], best_et_s * 1000.0


# Alias kept so external code that imports `FrequencyFirstSolver` still works
FrequencyFirstSolver = Alt4Solver


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
            self._solver = Alt4Solver(
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
                # Evict request with the lowest baseline reward (proxy).
                v_t = []
                for r in chosen:
                    t_q_s_r = per_request_time_ms(
                        self._latency, f_mhu, r.is_prefill, r.l_q, r.l_kv
                    ) / 1000.0
                    v = (
                        baseline_reward(r, self._cfg)
                        - self._cfg.beta * self._power.power_watts(f_mhu) * t_q_s_r
                    )
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
                    f"[energy_sched-alt4] iter={self._iter} f*={int(f_star)} "
                    f"|B|={len(chosen)} solve_ms={solve_ms:.2f} "
                    f"exec_ms={exec_str}",
                    flush=True,
                )
            return out

    return EnergyScheduler
