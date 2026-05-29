from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import math
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
    w_ttft: float = 10.0          # MUTABLE — initial value from env, drifts online
    w_tpot: float = 1.0          # MUTABLE — initial value from env, drifts online
    # Hardcoded learning rates for the SLO-weight online update (Sec.
    # "Adaptive Control"):
    #     w_TTFT ← [w_TTFT + eta_ttft · w_n · (TTFT_obs/TTFT_slo − 1)]^+
    #     w_TPOT ← [w_TPOT + eta_tpot · w_n · (avg_TPOT_obs/TPOT_slo − 1)]^+
    # Not exposed through env on purpose — main.sh need not change.
    eta_ttft: float = 0.0
    eta_tpot: float = 0.0
    eta_ms: float = 1e9          # accepted from env for backward compat; UNUSED
    Lmax: int = 0
    max_batch_size: int = 0      # 0 → inherit from vLLM scheduler_config.max_num_seqs
    default_w_n: float = 1.0
    default_ttft_ms: float = 4000.0
    default_tpot_ms: float = 200.0
    freq_candidates: Optional[List[int]] = None
    freq_stride: int = 1
    solution_mode: int = 1   # 1=H2 (freq-indep priority), 2=H3 (freq-dep priority)
    chunked_prefill: bool = False  # True → recognise partial-prefill in running queue
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
            solution_mode=int(os.environ.get("VLLM_ENERGY_SOLUTION_MODE", "1")),
            chunked_prefill=os.environ.get("VLLM_ENERGY_CHUNKED_PREFILL", "0") == "1",
            iter_log_path=os.environ.get("VLLM_ENERGY_ITER_LOG"),
        )


@dataclass
class ReqView:
    handle: Any
    is_prefill: bool
    l_q: int
    l_kv: int
    wait_ms: float        # input convention: ms
    deadline_ms: float    # input convention: ms (TTFT for prefill, TPOT for decode)
    w_n: float
    kv_blocks_needed: int = 0       # full KV size in blocks (mode 1)
    kv_blocks_incremental: int = 0  # new blocks needed this iter (mode 2)


# --- helper: baseline reward r_n -------------------------------------------

def baseline_reward(r: ReqView, cfg: EnergySchedConfig) -> float:
    """r_n = w_n · (w_TTFT for prefill, w_TPOT for decode)."""
    return r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)


# === The Alt-1 HEURISTIC solver ============================================

class Alt1HeuristicSolver:
    """Heuristic Alt-1 solver:
        Step 1  one-shot priority q_n = r_n · min(exp(−s_n), CAP) / ℓ_n
        Step 2  density-greedy fill until L_max or B_max binds  (→ B̂)
        Step 3  enumerate (B_j, f) ∈ {prefixes of B̂} × F jointly,
                pick (j*, f*) = argmax of the Alt-1 utility; final batch B_{j*}

    All time arithmetic in SECONDS; energy_model boundary is /1000 (input)
    and ×1000 (et_pred output for log compatibility).
    """

    # === HARDCODED HEURISTIC PARAMETER (so main.sh need not change) =========
    # CAP on exp(−s_n) in the priority q_n.  Chosen so that requests whose
    # slack is already < −ln(CAP) all share the same maximum "urgency boost"
    # for the priority order; the objective in Step 3 still uses the
    # uncapped exp.  EXP_CAP = 5 ⇒ s_n < −ln(5) ≈ −1.609 s saturates.
    EXP_CAP: float = 200000.0

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
        # Sort freqs ascending — irrelevant for correctness, mirrors alt3.
        self.freq_candidates = sorted(freq_candidates)

    def solve(
        self,
        reqs: List[ReqView],
        Lmax: int,
        Bmax: int,
        debug_iter: int = -1,
    ) -> Tuple[float, list, float]:
        if self.cfg.solution_mode == 2:
            return self._solve_h3(reqs, Lmax, Bmax, debug_iter)
        return self._solve_h2(reqs, Lmax, Bmax, debug_iter)

    def _solve_h2(
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

        # ---- (1) Vectorise per-request quantities (all time in s) --------
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
        # tok_arr = ℓ_{i,n}: per-iter token cost (prompt-len for prefill, 1 for decode).
        tok_arr = np.fromiter((r.l_q for r in reqs), dtype=np.int64, count=N)

        # r_n: baseline reward
        r_n_vec = w_n * np.where(is_pf, cfg.w_ttft, cfg.w_tpot)

        # s_n: slack in SECONDS  (= deadline_s − wait_s); s_n>0 ⇒ on time
        s_n_s = deadline_s - wait_s

        # Per-request workload contributions to (W_p, W_d) for batch_time.
        #   prefill : a_p·l_q² + b_p·l_q·l_kv + c_p·l_q   (added when n is prefill)
        #   decode  : a_d·l_kv  + b_d                    (added when n is decode)
        # NOTE the decode form mirrors energy_model.per_request_time_ms exactly.
        wp_contrib = np.where(
            is_pf, lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q, 0.0
        )
        wd_contrib = np.where(
            is_pf, 0.0, lat.a_d * l_kv + lat.b_d
        )

        # ---- (2) Step 1: priority q_n = r_n · min(exp(−s_n), CAP) / ℓ_n --
        # exp(−s_n) is computed in seconds (s_n in s, exponent dimensionless).
        # min(·, CAP) caps the boost from being deeply overdue; CAP = EXP_CAP.
        cap = float(self.EXP_CAP)
        urgency = np.minimum(np.exp(-s_n_s), cap)         # (N,)
        # Guard against ℓ_n=0 (shouldn't happen — prefill l_q≥1, decode l_q=1).
        ell_safe = np.maximum(tok_arr.astype(np.float64), 1.0)
        q_n = r_n_vec * urgency / ell_safe                 # (N,)

        # ---- (3) Step 2: density-greedy fill -----------------------------
        order = np.argsort(-q_n, kind="stable")            # descending

        B_eff = int(Bmax) if Bmax > 0 else N
        used_tok = 0
        picked_local: List[int] = []
        W_p = 0.0
        W_d = 0.0
        has_pf = False
        has_dc = False

        chunked = self.cfg.chunked_prefill
        for idx in order:
            if len(picked_local) >= B_eff:
                break  # |B| cap binds — no further additions possible
            tok_n = int(tok_arr[idx])
            if used_tok + tok_n > Lmax:
                if chunked and bool(is_pf[idx]):
                    tok_n = Lmax - used_tok
                    if tok_n <= 0:
                        continue
                    actual_lq = float(tok_n)
                    actual_lkv = float(l_kv[idx])
                    tok_arr[idx] = tok_n
                    wp_contrib[idx] = (lat.a_p * actual_lq * actual_lq
                                       + lat.b_p * actual_lq * actual_lkv
                                       + lat.c_p * actual_lq)
                    reqs[int(idx)].l_q = tok_n
                else:
                    continue
            picked_local.append(int(idx))
            used_tok += tok_n
            if bool(is_pf[idx]):
                has_pf = True
                W_p += float(wp_contrib[idx])
            else:
                has_dc = True
                W_d += float(wd_contrib[idx])

        if not picked_local:
            return float(default_f), [], 0.0

        # ---- (4) Step 3: joint (B_j, f) enumeration over prefixes -------
        # Build cumulative workload sums over picked_local in admission
        # order.  Because {B_j} is a nested chain (B_1 ⊂ B_2 ⊂ ... ⊂ B̂),
        # ET_i(B_j, f) for all j is recovered "for free" via cumsum once
        # we have the per-item contributions.  All time in SECONDS.
        K = len(picked_local)                                       # |B̂|
        picked_idx = np.asarray(picked_local, dtype=np.int64)       # (K,)
        is_pf_picked = is_pf[picked_idx]                            # (K,)
        # Per-item prefill/decode workload contributions (already 0 for
        # the wrong mode by construction in wp_contrib / wd_contrib).
        dp = wp_contrib[picked_idx].astype(np.float64)              # (K,)
        dd = wd_contrib[picked_idx].astype(np.float64)              # (K,)
        cum_dp = np.cumsum(dp)                                      # (K,)
        cum_dd = np.cumsum(dd)                                      # (K,)
        # Mode indicators are monotone non-decreasing along the prefix
        # chain: once any prefill (resp. decode) is admitted, I_p=1
        # (resp. I_d=1) for all longer prefixes.
        cum_has_pf = (np.cumsum(is_pf_picked.astype(np.int64)) > 0).astype(np.float64)
        cum_has_dc = (np.cumsum((~is_pf_picked).astype(np.int64)) > 0).astype(np.float64)
        # Numerators of the ET formula, per prefix.  Units: ms·MHz scale,
        # consistent with energy_model.batch_overhead_ms; the final /1000
        # converts to seconds at the bottom.
        num_p_vec = cum_dp + lat.w_pf * cum_has_pf                  # (K,)
        num_d_vec = cum_dd + lat.w_dec * cum_has_dc                 # (K,)

        # Frequency candidates (subsampled by stride).
        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates
        f_arr = np.asarray(freqs, dtype=np.float64)                 # (|F|,)
        f_alpha_arr = f_arr ** lat.alpha                            # (|F|,)
        F = f_arr.size

        # ET_s_mat[fi, j-1] = ET_i(B_j, f_arr[fi]) in SECONDS
        # = (num_p[j-1]/f + num_d[j-1]/f^α + t_c) / 1000
        ET_s_mat = (
            num_p_vec[None, :] / f_arr[:, None]                     # (|F|, K)
            + num_d_vec[None, :] / f_alpha_arr[:, None]
            + lat.t_c
        ) / 1000.0                                                   # (|F|, K)

        # Per-picked-request s_n and r_n in admission order.  All seconds.
        s_picked = s_n_s[picked_idx]                                # (K,)
        r_picked = r_n_vec[picked_idx]                              # (K,)

        # u_mat[fi, j-1, n] = r_picked[n] · exp(−[ET_s_mat[fi, j-1] − s_picked[n]]_+)
        # Memory: |F| × K² × 8 bytes; for |F|=21, K=128 ≈ 2.7 MB.
        overshoot = (
            ET_s_mat[:, :, None]                                    # (|F|, K, 1)
            - s_picked[None, None, :]                               # (1, 1, K)
        )                                                            # (|F|, K, K)
        np.maximum(overshoot, 0.0, out=overshoot)
        u_mat = r_picked[None, None, :] * np.exp(-overshoot)        # (|F|, K, K)

        # Lower-triangular mask: for prefix B_j, only items with index n
        # in [0, j-1] (admission order) belong to B_j.  tri[j-1, n] = 1
        # iff n ≤ j-1, i.e. n < j.
        tri = np.tri(K, K, dtype=np.float64)                        # (K, K)
        sum_u_per_jf = (u_mat * tri[None, :, :]).sum(axis=2)        # (|F|, K)

        # Energy term in JOULES = β · P(f) · ET_s.
        P_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )                                                            # (|F|,)
        J_mat = sum_u_per_jf - beta * P_arr[:, None] * ET_s_mat     # (|F|, K)

        # ---- Plan A: ALWAYS commit; argmax over (j, f), J* may be < 0 ---
        flat_idx = int(np.argmax(J_mat))
        best_fi, best_jidx = np.unravel_index(flat_idx, J_mat.shape)
        best_j_size = int(best_jidx) + 1                            # |B_{j*}|
        best_f = float(f_arr[best_fi])
        best_J = float(J_mat[best_fi, best_jidx])
        best_et_s = float(ET_s_mat[best_fi, best_jidx])
        chosen_local = picked_local[:best_j_size]

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in chosen_local if reqs[i].is_prefill)
            n_d_ch = best_j_size - n_p_ch
            print(
                f"[dbg-alg3_1] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"greedy=B̂[{K}] chosen=B_{best_j_size}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} J*={best_J:.3f} ET={best_et_s*1000.0:.2f}ms "
                f"B_max={B_eff} CAP={self.EXP_CAP}",
                file=sys.stderr, flush=True)

        # Convert ET back to ms for compatibility with the iter_log schema.
        return float(best_f), [reqs[i] for i in chosen_local], best_et_s * 1000.0

    # === Heuristic 3: freq-dependent priority, per-frequency admission ======

    def _solve_h3(
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
        cap = float(self.EXP_CAP)

        # ---- Vectorise per-request quantities (all time in s) ---------------
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

        r_n_vec = w_n * np.where(is_pf, cfg.w_ttft, cfg.w_tpot)
        s_n_s = deadline_s - wait_s
        urgency = np.minimum(np.exp(-s_n_s), cap)
        ell_safe = np.maximum(tok_arr.astype(np.float64), 1.0)

        wp_contrib = np.where(
            is_pf, lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q, 0.0
        )
        wd_contrib = np.where(is_pf, 0.0, lat.a_d * l_kv + lat.b_d)

        B_eff = int(Bmax) if Bmax > 0 else N

        # Frequency candidates (subsampled by stride).
        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates

        # ==== Batch vectorisation: Step 1 + Sort for ALL frequencies ========
        f_arr = np.asarray(freqs, dtype=np.float64)            # (F,)
        F = f_arr.size
        f_alpha_arr = f_arr ** lat.alpha                        # (F,)
        P_f_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )                                                        # (F,)
        RU = r_n_vec * urgency                                  # (N,)
        # wp_contrib=0 for decode, wd_contrib=0 for prefill → sum is correct
        t_nf_all = (
            wp_contrib[None, :] / f_arr[:, None]
            + wd_contrib[None, :] / f_alpha_arr[:, None]
        ) / 1000.0                                               # (F, N)
        q_all = (
            RU[None, :] - beta * P_f_arr[:, None] * t_nf_all
        ) / ell_safe[None, :]                                    # (F, N)
        orders_all = np.argsort(-q_all, axis=1, kind="stable")   # (F, N)

        # ==== Precompute exp(s_n) for exp-decomposition in Step 3 ===========
        exp_s_all = np.exp(s_n_s)                               # (N,)

        # ---- Frequency loop (Steps 2–3 only) -------------------------------
        global_best_J = -np.inf
        global_best_f = float(default_f)
        global_best_chosen: List[int] = []
        global_best_et_s = 0.0
        global_best_chunked: dict = {}

        chunked = self.cfg.chunked_prefill
        for fi in range(F):
            f_float = float(f_arr[fi])
            f_alpha = float(f_alpha_arr[fi])
            P_f = float(P_f_arr[fi])
            order_fi = orders_all[fi]                            # (N,)

            # -- Step 2: greedy admission (fast-path via cumsum) -------------
            # chunked_override: maps position-in-picked → (new_wp, actual_lq)
            # (only set when a prefill request is chunked to fit Lmax).
            chunked_override: dict = {}
            tok_sorted = tok_arr[order_fi]                       # (N,)
            B_try = min(B_eff, N)
            cum_tok_b = np.cumsum(tok_sorted[:B_try])
            if cum_tok_b[B_try - 1] <= Lmax:
                picked_local = order_fi[:B_try].tolist()
            else:
                # Fallback: sequential scan with skip/chunk semantics
                order_list = order_fi.tolist()
                tok_list = tok_sorted.tolist()
                picked_local = []
                used_tok = 0
                for i in range(N):
                    if len(picked_local) >= B_eff:
                        break
                    tok_n = tok_list[i]
                    orig_idx = order_list[i]
                    if used_tok + tok_n > Lmax:
                        if chunked and bool(is_pf[orig_idx]):
                            tok_n = Lmax - used_tok
                            if tok_n <= 0:
                                continue
                            actual_lq = float(tok_n)
                            actual_lkv = float(l_kv[orig_idx])
                            new_wp = (lat.a_p * actual_lq * actual_lq
                                      + lat.b_p * actual_lq * actual_lkv
                                      + lat.c_p * actual_lq)
                            chunked_override[len(picked_local)] = (new_wp, tok_n)
                        else:
                            continue
                    picked_local.append(orig_idx)
                    used_tok += tok_n

            if not picked_local:
                continue

            # -- Step 3: prefix maximization at fixed f -----------------------
            K = len(picked_local)
            picked_idx = np.asarray(picked_local, dtype=np.int64)
            is_pf_picked = is_pf[picked_idx]

            dp = wp_contrib[picked_idx].astype(np.float64)
            if chunked_override:
                for pos, (new_wp, _) in chunked_override.items():
                    dp[pos] = new_wp
            dd = wd_contrib[picked_idx].astype(np.float64)
            cum_dp = np.cumsum(dp)
            cum_dd = np.cumsum(dd)
            cum_has_pf = (np.cumsum(is_pf_picked.astype(np.int64)) > 0).astype(np.float64)
            cum_has_dc = (np.cumsum((~is_pf_picked).astype(np.int64)) > 0).astype(np.float64)

            num_p_vec = cum_dp + lat.w_pf * cum_has_pf
            num_d_vec = cum_dd + lat.w_dec * cum_has_dc

            ET_s = (num_p_vec / f_float + num_d_vec / f_alpha + lat.t_c) / 1000.0

            r_picked = r_n_vec[picked_idx]

            # Exp decomposition: exp(-max(ET_j - s_n, 0)) = min(exp(s_n)*exp(-ET_j), 1)
            exp_s = exp_s_all[picked_idx]                        # (K,)
            exp_neg_ET = np.exp(-ET_s)                           # (K,)
            raw = exp_neg_ET[:, None] * exp_s[None, :]           # (K, K)
            u_mat = r_picked[None, :] * np.minimum(raw, 1.0)    # (K, K)
            tri = np.tri(K, K, dtype=np.float64)
            sum_u = (u_mat * tri).sum(axis=1)                    # (K,)

            J_vec = sum_u - beta * P_f * ET_s                    # (K,)
            best_jidx = int(np.argmax(J_vec))
            best_J_f = float(J_vec[best_jidx])

            if best_J_f > global_best_J:
                global_best_J = best_J_f
                global_best_f = f_float
                global_best_chosen = picked_local[:best_jidx + 1]
                global_best_et_s = float(ET_s[best_jidx])
                global_best_chunked = dict(chunked_override)

        if not global_best_chosen:
            return float(default_f), [], 0.0

        n_chosen = len(global_best_chosen)
        for pos, (_, actual_lq) in global_best_chunked.items():
            if pos < n_chosen:
                reqs[global_best_chosen[pos]].l_q = actual_lq

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in global_best_chosen if reqs[i].is_prefill)
            n_d_ch = len(global_best_chosen) - n_p_ch
            print(
                f"[dbg-h3] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"chosen=B_{len(global_best_chosen)}(p={n_p_ch}d={n_d_ch}) "
                f"f={global_best_f:.0f} J*={global_best_J:.3f} "
                f"ET={global_best_et_s*1000.0:.2f}ms "
                f"B_max={B_eff} CAP={self.EXP_CAP}",
                file=sys.stderr, flush=True)

        return float(global_best_f), [reqs[i] for i in global_best_chosen], global_best_et_s * 1000.0


# Alias kept so external code that imports `FrequencyFirstSolver` still works
FrequencyFirstSolver = Alt1HeuristicSolver


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
            if cands[-1] != 1410:
                cands.append(1410)
            #cands = [1410]
            self._solver = Alt1HeuristicSolver(
                self._cfg, self._latency, self._power, cands
            )
            if self._cfg.Lmax <= 0:
                self._cfg.Lmax = int(getattr(
                    self.scheduler_config, "max_num_batched_tokens",
                    getattr(self.scheduler_config, "max_model_len", 8192),
                ))
            self._active_cap = int(getattr(
                self.scheduler_config, "max_num_seqs", 128
            ))
            if self._cfg.max_batch_size <= 0:
                self._cfg.max_batch_size = self._active_cap
            self._iter_log = _open_iter_log(self._cfg.iter_log_path)
            self._prev_exit_t = None
            self._iter = 0
            self._prev_record = None
            self._last_exec = {}
            self._materialise_fail_streak = 0
            # Per-request state for the SLO-weight online update.
            # Keyed by req_id, value is a dict with the following fields:
            #   arrival_ms      : float, request arrival time (ms wall-clock)
            #   w_n             : float, request priority (cached so we keep
            #                     it after the request leaves self.running)
            #   ttft_slo_ms     : float, per-request TTFT SLO (deadline)
            #   tpot_slo_ms     : float, per-request TPOT SLO (deadline)
            #   ttft_fired      : bool,  has the TTFT update fired for this req?
            #   first_tok_ms    : Optional[float], wall-clock of first token
            #                     (set when ttft_fired transitions to True)
            #   last_num_out    : int,   last-observed num_output_tokens
            self._req_state: dict = {}

        def _build_request_views(self, now_ms: float,
                                    skip_waiting: bool = False,
                                    ) -> List[ReqView]:
            reqs: List[ReqView] = []
            block_size = getattr(self, "block_size", 16)
            chunked = self._cfg.chunked_prefill
            waiting_iter = () if skip_waiting else self.waiting
            for req in waiting_iter:
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
                arrival_ms = self._get_arrival_ms(req, now_ms)
                wait_ms = now_ms - arrival_ms
                l_q = getattr(req, "num_prompt_tokens", 0)
                l_kv = 0
                kv_blocks = (l_q + block_size - 1) // block_size
                reqs.append(ReqView(
                    handle=req, is_prefill=True, l_q=l_q, l_kv=l_kv,
                    wait_ms=wait_ms, deadline_ms=ttft, w_n=w_n,
                    kv_blocks_needed=kv_blocks,
                    kv_blocks_incremental=kv_blocks,
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
                num_prompt = getattr(req, "num_prompt_tokens", 0)
                num_computed = getattr(req, "num_computed_tokens", 0)
                if chunked and num_computed < num_prompt:
                    # --- partial prefill: still in prefill phase ---
                    remaining = num_prompt - num_computed
                    l_q = remaining
                    l_kv = num_computed
                    arrival_ms = self._get_arrival_ms(req, now_ms)
                    wait_ms = now_ms - arrival_ms
                    kv_blocks = (num_prompt + block_size - 1) // block_size
                    kv_inc = ((num_computed + remaining + block_size - 1) // block_size
                              - (num_computed + block_size - 1) // block_size)
                    reqs.append(ReqView(
                        handle=req, is_prefill=True, l_q=l_q, l_kv=l_kv,
                        wait_ms=wait_ms, deadline_ms=ttft, w_n=w_n,
                        kv_blocks_needed=kv_blocks,
                        kv_blocks_incremental=kv_inc,
                    ))
                else:
                    # --- decode (original logic) ---
                    req_id = getattr(req, "request_id", id(req))
                    last_exec_ms = self._last_exec.get(req_id)
                    if last_exec_ms is not None:
                        wait_ms = now_ms - last_exec_ms
                    else:
                        wait_ms = now_ms - self._get_arrival_ms(req, now_ms)
                    l_kv = num_computed
                    l_q = 1
                    kv_blocks = (l_kv + block_size) // block_size
                    kv_inc = ((l_kv + 1 + block_size - 1) // block_size
                              - (l_kv + block_size - 1) // block_size)
                    reqs.append(ReqView(
                        handle=req, is_prefill=False, l_q=l_q, l_kv=l_kv,
                        wait_ms=wait_ms, deadline_ms=tpot, w_n=w_n,
                        kv_blocks_needed=kv_blocks,
                        kv_blocks_incremental=kv_inc,
                    ))
            return reqs

        def _enforce_active_cap(
            self, chosen: List[ReqView], now_ms: float,
        ) -> Tuple[List[ReqView], int]:
            """Ensure len(running) + newly_admitted <= active_cap.

            Phase 1: preempt non-chosen running (lowest num_computed_tokens first).
            Phase 2: drop least-urgent new admissions from chosen.
            """
            waiting_set = set(self.waiting)
            n_new = sum(1 for r in chosen if r.handle in waiting_set)
            active_after = len(self.running) + n_new

            if active_after <= self._active_cap:
                return chosen, 0

            # Phase 1: preempt non-chosen running requests
            chosen_handles = {r.handle for r in chosen}
            victims = [
                req for req in self.running
                if req not in chosen_handles
            ]
            victims.sort(
                key=lambda r: getattr(r, "num_computed_tokens", 0)
            )
            n_preempted = 0
            timestamp_s = now_ms / 1000.0
            for req in victims:
                if active_after <= self._active_cap:
                    break
                rid = getattr(req, "request_id", id(req))
                self.running.remove(req)
                self._preempt_request(req, timestamp_s)
                self._last_exec.pop(rid, None)
                self._req_state.pop(rid, None)
                n_preempted += 1
                active_after -= 1

            # Phase 2: if still over, drop least-urgent new admissions from chosen
            if active_after > self._active_cap:
                new_reqs = [r for r in chosen if r.handle in waiting_set]
                new_reqs.sort(key=lambda r: r.deadline_ms - r.wait_ms, reverse=True)
                to_remove = set()
                for r in new_reqs:
                    if active_after <= self._active_cap:
                        break
                    to_remove.add(id(r))
                    active_after -= 1
                chosen = [r for r in chosen if id(r) not in to_remove]

            return chosen, n_preempted

        def _kv_evict(
            self, chosen: List[ReqView], f_mhu: float,
            now_ms: float = 0.0,
        ) -> Tuple[List[ReqView], dict]:
            """Ensure enough free KV blocks for chosen's incremental needs.

            Three-phase eviction (all sorted by num_computed_tokens ascending):
              Phase A: drop waiting requests from chosen (reduces demand, zero cost)
              Phase B: preempt unchosen running (frees supply, no solver disruption)
              Phase C: preempt chosen running (frees supply, last resort)

            Returns (chosen, evict_info_dict) where evict_info_dict contains
            detailed counts for logging.
            """
            info = {
                "n_dropped_waiting_kv": 0,
                "n_preempted_kv_unchosen": 0,
                "n_preempted_kv_chosen": 0,
                "free_blocks_before": 0,
                "kv_needed": 0,
                "free_blocks_after": 0,
            }
            kv_mgr = getattr(self, "kv_cache_manager", None)
            if kv_mgr is None:
                return chosen, info
            block_pool = getattr(kv_mgr, "block_pool", None)
            if block_pool is None:
                return chosen, info
            free_fn = getattr(block_pool, "get_num_free_blocks", None)
            if free_fn is None:
                return chosen, info

            def _total_needed():
                return sum(r.kv_blocks_incremental for r in chosen)

            needed = _total_needed()
            free_before = free_fn()
            info["free_blocks_before"] = free_before
            info["kv_needed"] = needed

            if needed <= free_before:
                info["free_blocks_after"] = free_before
                return chosen, info

            waiting_set = set(self.waiting)
            timestamp_s = now_ms / 1000.0
            needed_remaining = needed

            # --- Phase A: drop waiting requests from chosen ---
            # (reduces demand; they have no KV yet, cheapest action)
            # Never drop the last request — let Phase B/C free KV instead.
            chosen_waiting = [
                r for r in chosen if r.handle in waiting_set
            ]
            chosen_waiting.sort(
                key=lambda r: getattr(r.handle, "num_computed_tokens", 0)
            )
            n_non_waiting = sum(1 for r in chosen if r.handle not in waiting_set)
            to_drop = set()
            for r in chosen_waiting:
                if needed_remaining <= free_fn():
                    break
                if len(chosen) - len(to_drop) <= 1:
                    break
                if n_non_waiting == 0 and len(chosen_waiting) - len(to_drop) <= 1:
                    break
                needed_remaining -= r.kv_blocks_incremental
                to_drop.add(id(r))
                info["n_dropped_waiting_kv"] += 1
            if to_drop:
                chosen = [r for r in chosen if id(r) not in to_drop]

            if needed_remaining <= free_fn():
                info["free_blocks_after"] = free_fn()
                return chosen, info

            # --- Phase B: preempt unchosen running ---
            # (frees existing KV blocks; does not disrupt solver's batch)
            chosen_handles = {r.handle for r in chosen}
            unchosen_running = [
                req for req in self.running
                if req not in chosen_handles
            ]
            unchosen_running.sort(
                key=lambda r: getattr(r, "num_computed_tokens", 0)
            )
            for req in unchosen_running:
                if needed_remaining <= free_fn():
                    break
                rid = getattr(req, "request_id", id(req))
                self.running.remove(req)
                self._preempt_request(req, timestamp_s)
                self._last_exec.pop(rid, None)
                self._req_state.pop(rid, None)
                info["n_preempted_kv_unchosen"] += 1

            if needed_remaining <= free_fn():
                info["free_blocks_after"] = free_fn()
                return chosen, info

            # --- Phase C: preempt chosen running (last resort) ---
            # (frees existing KV blocks; breaks solver decision)
            chosen_running = [
                r for r in chosen if r.handle not in waiting_set
            ]
            chosen_running.sort(
                key=lambda r: getattr(r.handle, "num_computed_tokens", 0)
            )
            to_preempt = set()
            for r in chosen_running:
                if needed_remaining <= free_fn():
                    break
                rid = getattr(r.handle, "request_id", id(r.handle))
                self.running.remove(r.handle)
                self._preempt_request(r.handle, timestamp_s)
                self._last_exec.pop(rid, None)
                self._req_state.pop(rid, None)
                needed_remaining -= r.kv_blocks_incremental
                to_preempt.add(id(r))
                info["n_preempted_kv_chosen"] += 1
            if to_preempt:
                chosen = [r for r in chosen if id(r) not in to_preempt]

            info["free_blocks_after"] = free_fn()
            return chosen, info

        def _materialise_batch(self, chosen: List[ReqView]):
            if self._cfg.chunked_prefill:
                waiting_set = set(self.waiting)
                chosen_handles = {r.handle for r in chosen}
                waiting_handles = {h for h in chosen_handles if h in waiting_set}
                running_handles = {h for h in chosen_handles if h not in waiting_set}
            else:
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

        def _extract_slos(self, req) -> Tuple[float, float, float]:
            """Pull (ttft_slo_ms, tpot_slo_ms, w_n) from a vLLM Request handle.
            Falls back to EnergySchedConfig defaults when extra_args is missing.
            """
            sp = getattr(req, "sampling_params", None)
            ea = getattr(sp, "extra_args", {}) if sp else {}
            if not isinstance(ea, dict):
                ea = {}
            ttft = float(ea.get("ttft_ms", self._cfg.default_ttft_ms))
            tpot = float(ea.get("tpot_ms", self._cfg.default_tpot_ms))
            w_n = float(ea.get("w_n", self._cfg.default_w_n))
            return ttft, tpot, w_n

        def _get_arrival_ms(self, req, now_ms: float) -> float:
            """Return the best-known arrival time in ms for *req*.

            Priority: client-side send_time (from extra_args) > vLLM
            req.arrival_time > now_ms fallback.  send_time is the wall-clock
            epoch (seconds) recorded by workload_sender right before the HTTP
            POST, so it captures the full queuing delay that vLLM's
            req.arrival_time may miss (the latter is set inside
            input_processor.process_inputs, which can be delayed when the
            engine event loop is busy).
            """
            sp = getattr(req, "sampling_params", None)
            ea = getattr(sp, "extra_args", {}) if sp else {}
            if isinstance(ea, dict):
                st = ea.get("send_time")
                if st is not None:
                    return float(st) * 1000.0
            if hasattr(req, "arrival_time") and req.arrival_time is not None:
                return float(req.arrival_time) * 1000.0
            return now_ms

        def _ensure_req_state(self, rid, req, now_ms: float):
            """Initialise self._req_state[rid] on first sight of the request."""
            if rid in self._req_state:
                return
            ttft_slo, tpot_slo, w_n = self._extract_slos(req)
            arrival_ms = self._get_arrival_ms(req, now_ms)
            self._req_state[rid] = {
                "arrival_ms": arrival_ms,
                "w_n": w_n,
                "ttft_slo_ms": ttft_slo,
                "tpot_slo_ms": tpot_slo,
                "ttft_fired": False,
                "first_tok_ms": None,
                "last_num_out": 0,
            }

        def _online_update_weights(self, now_ms: float) -> Tuple[int, int]:
            """Online update of w_TTFT and w_TPOT (Adaptive Control rule).

            Two events trigger updates:
              (TTFT)  Upon observing the first output token of request n:
                  w_TTFT ← [w_TTFT + eta_ttft · w_n · (TTFT_obs/TTFT_slo − 1)]^+
                  where TTFT_obs = now_ms − arrival_ms.
              (TPOT)  Upon completion of request n (request leaves visibility):
                  w_TPOT ← [w_TPOT + eta_tpot · w_n · (avg_TPOT_obs/TPOT_slo − 1)]^+
                  where avg_TPOT_obs = (now_ms − first_tok_ms) / (num_out − 1).

            Multiple events in the same iter are applied SERIALLY, in the
            order they are detected (each subsequent update sees the
            already-updated weight).  Returns the per-iter event counts
            (n_ttft_updates, n_tpot_updates) for logging.
            """
            cfg = self._cfg

            # --- Snapshot currently visible requests ---------------------
            visible_ids = set()
            running_view: List[tuple] = []  # [(rid, req), ...]
            for req in self.running:
                rid = getattr(req, "request_id", id(req))
                visible_ids.add(rid)
                running_view.append((rid, req))
                self._ensure_req_state(rid, req, now_ms)
            for req in self.waiting:
                rid = getattr(req, "request_id", id(req))
                visible_ids.add(rid)
                self._ensure_req_state(rid, req, now_ms)

            # --- Phase A: TTFT detection (serial accumulation) -----------
            n_ttft = 0
            for rid, req in running_view:
                st = self._req_state[rid]
                n_out = int(getattr(req, "num_output_tokens", 0))
                # Always refresh last_num_out so completion-time TPOT can use it
                st["last_num_out"] = n_out
                if n_out >= 1 and not st["ttft_fired"]:
                    ttft_obs_ms = now_ms - st["arrival_ms"]
                    slo = max(st["ttft_slo_ms"], 1e-6)
                    delta = cfg.eta_ttft * st["w_n"] * (ttft_obs_ms / slo - 1.0)
                    cfg.w_ttft = max(0.0, cfg.w_ttft + delta)  # [·]^+
                    #cfg.w_ttft = cfg.w_ttft * math.exp(delta)  # alternative multiplicative update
                    st["ttft_fired"] = True
                    st["first_tok_ms"] = now_ms
                    n_ttft += 1

            # --- Phase B: completion detection → TPOT update -------------
            # A req that was tracked but is no longer in self.running ∪
            # self.waiting has finished or been preempted-and-dropped.
            n_tpot = 0
            disappeared = list(set(self._req_state.keys()) - visible_ids)
            for rid in disappeared:
                st = self._req_state.pop(rid)
                if not st["ttft_fired"] or st["first_tok_ms"] is None:
                    # Never produced a first token → no TPOT samples.
                    continue
                n_decode_tokens = max(0, st["last_num_out"] - 1)
                if n_decode_tokens <= 0:
                    # Only produced first token — no inter-token interval.
                    continue
                avg_tpot_obs_ms = (now_ms - st["first_tok_ms"]) / n_decode_tokens
                slo = max(st["tpot_slo_ms"], 1e-6)
                delta = cfg.eta_tpot * st["w_n"] * (avg_tpot_obs_ms / slo - 1.0)
                cfg.w_tpot = max(0.01, cfg.w_tpot + delta)  # [·]^+
                #cfg.w_tpot = cfg.w_tpot * math.exp(delta)
                n_tpot += 1

            return n_ttft, n_tpot

        def schedule(self):
            t_enter = time.monotonic()
            exec_ms = (
                (t_enter - self._prev_exit_t) * 1000.0
                if self._prev_exit_t is not None else None
            )
            now_ms = time.time() * 1000.0
            # ----- Online SLO-weight update (Adaptive Control) ----------
            # Must run BEFORE solver.solve() so the fresh w_TTFT/w_TPOT
            # are picked up via cfg.w_ttft / cfg.w_tpot in the solver.
            n_ttft_upd, n_tpot_upd = self._online_update_weights(now_ms)
            # Snapshot post-update weights for the iter_log record below.
            w_ttft_now = float(self._cfg.w_ttft)
            w_tpot_now = float(self._cfg.w_tpot)

            skip_w = self._materialise_fail_streak >= 2 and len(self.running) > 0
            reqs = self._build_request_views(now_ms, skip_waiting=skip_w)
            # Capture queue lengths BEFORE _materialise_batch mutates them.
            len_waiting_before = len(self.waiting)
            len_running_before = len(self.running)
            t_solve0 = time.monotonic()
            f_star, chosen, et_pred = self._solver.solve(
                reqs, self._cfg.Lmax, self._cfg.max_batch_size, self._iter
            )
            solve_ms = (time.monotonic() - t_solve0) * 1000.0
            self._freq_ctl.set_frequency(int(f_star))
            n_preempted = 0
            n_preempted_a = 0
            kv_evict_info = {
                "n_dropped_waiting_kv": 0,
                "n_preempted_kv_unchosen": 0,
                "n_preempted_kv_chosen": 0,
                "free_blocks_before": 0,
                "kv_needed": 0,
                "free_blocks_after": 0,
            }
            # Snapshot solver's choice (req_ids) so we can diff against
            # what super().schedule() actually scheduled.
            chosen_prefill_ids: set = set()
            chosen_decode_ids: set = set()
            chosen_total_tokens = 0
            for r in chosen:
                rid = getattr(r.handle, "request_id", str(id(r.handle)))
                if r.is_prefill:
                    chosen_prefill_ids.add(rid)
                else:
                    chosen_decode_ids.add(rid)
                chosen_total_tokens += int(r.l_q)
            if not chosen:
                out = super().schedule()
            else:
                # Sync kv_blocks_incremental for chunked prefills (solver may
                # have reduced r.l_q to a chunk smaller than the full remaining).
                if self._cfg.chunked_prefill:
                    block_size = getattr(self, "block_size", 16)
                    for r in chosen:
                        if r.is_prefill:
                            new_end = int(r.l_kv) + int(r.l_q)
                            r.kv_blocks_incremental = (
                                (new_end + block_size - 1) // block_size
                                - (int(r.l_kv) + block_size - 1) // block_size
                            )
                chosen, n_preempted_a = self._enforce_active_cap(chosen, now_ms)
                chosen, kv_evict_info = self._kv_evict(chosen, f_star, now_ms)
                n_preempted_k = (kv_evict_info["n_preempted_kv_unchosen"]
                                 + kv_evict_info["n_preempted_kv_chosen"])
                n_preempted = n_preempted_a + n_preempted_k
                if not chosen:
                    out = super().schedule()
                else:
                    out = self._materialise_batch(chosen)
                    if out.total_num_scheduled_tokens == 0:
                        self._materialise_fail_streak += 1
                        out = super().schedule()
                        chosen = []
                        for rid in out.num_scheduled_tokens:
                            self._last_exec[rid] = now_ms
                    else:
                        self._materialise_fail_streak = 0
                        for r in chosen:
                            req_id = getattr(r.handle, "request_id", id(r.handle))
                            self._last_exec[req_id] = now_ms
            # Per-batch composition counts (what solver chose AFTER _kv_evict).
            n_prefill = sum(1 for r in chosen if r.is_prefill)
            n_decode = len(chosen) - n_prefill

            # ---- actual-vs-chosen diagnostics --------------------------
            # Extract what the parent scheduler actually scheduled this step.
            actual_new_ids = {nr.req_id for nr in out.scheduled_new_reqs}
            cached = out.scheduled_cached_reqs
            actual_cached_ids = set(cached.req_ids)
            resumed_ids = set(cached.resumed_req_ids)
            actual_prefill_ids = actual_new_ids | resumed_ids
            actual_decode_ids = actual_cached_ids - resumed_ids
            # Reclassify partial-prefill running requests: solver marks them
            # as prefill (still doing prefill work), but parent reports them
            # as cached (they are in the running queue).  Align with solver.
            partial_pf = actual_cached_ids & chosen_prefill_ids
            actual_prefill_ids = actual_prefill_ids | partial_pf
            actual_decode_ids = actual_decode_ids - partial_pf
            parent_preempted_ids = set(out.preempted_req_ids or ())
            # Diff against solver's choice (only meaningful when chosen is non-empty).
            dropped_prefill_ids = chosen_prefill_ids - actual_prefill_ids
            dropped_decode_ids = chosen_decode_ids - actual_decode_ids
            extra_prefill_ids = actual_prefill_ids - chosen_prefill_ids
            extra_decode_ids = actual_decode_ids - chosen_decode_ids

            # High-visibility warning when parent silently drops solver picks.
            if chosen and (dropped_prefill_ids or dropped_decode_ids):
                print(
                    f"[energy_sched-MISMATCH] iter={self._iter} "
                    f"chosen=p{len(chosen_prefill_ids)}d{len(chosen_decode_ids)} "
                    f"actual=p{len(actual_prefill_ids)}d{len(actual_decode_ids)} "
                    f"dropped=p{len(dropped_prefill_ids)}d{len(dropped_decode_ids)} "
                    f"extra=p{len(extra_prefill_ids)}d{len(extra_decode_ids)} "
                    f"parent_preempted={len(parent_preempted_ids)} "
                    f"len_w={len_waiting_before} len_r={len_running_before}",
                    flush=True,
                )

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
                        "n_prefill": n_prefill,
                        "n_decode": n_decode,
                        "n_preempted": n_preempted,
                        "n_preempted_active_cap": n_preempted_a,
                        "n_preempted_kv_unchosen": kv_evict_info["n_preempted_kv_unchosen"],
                        "n_preempted_kv_chosen": kv_evict_info["n_preempted_kv_chosen"],
                        "n_dropped_waiting_kv": kv_evict_info["n_dropped_waiting_kv"],
                        "free_blocks_before": kv_evict_info["free_blocks_before"],
                        "kv_needed": kv_evict_info["kv_needed"],
                        "free_blocks_after": kv_evict_info["free_blocks_after"],
                        "f_star": int(f_star),
                        "et_pred_ms": et_pred,
                        "w_ttft": w_ttft_now,
                        "w_tpot": w_tpot_now,
                        "ttft_updates": n_ttft_upd,
                        "tpot_updates": n_tpot_upd,
                        "len_waiting_before": len_waiting_before,
                        "len_running_before": len_running_before,
                        "chosen_total_tokens": chosen_total_tokens,
                        "actual_prefill": len(actual_prefill_ids),
                        "actual_decode": len(actual_decode_ids),
                        "actual_total_tokens": int(out.total_num_scheduled_tokens),
                        "dropped_prefill": len(dropped_prefill_ids),
                        "dropped_decode": len(dropped_decode_ids),
                        "extra_prefill": len(extra_prefill_ids),
                        "extra_decode": len(extra_decode_ids),
                        "parent_preempted": len(parent_preempted_ids),
                    }
                else:
                    self._prev_record = None
                self._iter += 1
            self._prev_exit_t = time.monotonic()
            if self._iter_log is not None and self._iter % self._cfg.log_every_n == 0:
                exec_str = f"{exec_ms:.2f}" if exec_ms else "N/A"
                print(
                    f"[energy_sched-alg3_1] iter={self._iter} f*={int(f_star)} "
                    f"|B|={len(chosen)}(p={n_prefill}d={n_decode}) "
                    f"actual(p{len(actual_prefill_ids)}d{len(actual_decode_ids)}) "
                    f"solve_ms={solve_ms:.2f} exec_ms={exec_str} "
                    f"w_ttft={w_ttft_now:.3f} w_tpot={w_tpot_now:.3f} "
                    f"upd=({n_ttft_upd},{n_tpot_upd})",
                    flush=True,
                )
            return out

    return EnergyScheduler
