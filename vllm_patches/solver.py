"""Pure-algorithm solver for energy-efficient LLM scheduling.

No vLLM imports — only numpy + stdlib.  This module is copied into
vllm/energy_sched/ by apply_patch.sh and consumed by the energy branch
inside vllm.v1.core.sched.scheduler.Scheduler.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .energy_model import (
    LatencyParams,
    PowerParams,
    per_request_time_ms,
    batch_overhead_ms,
    load_latency_params,
    load_power_params,
)


@dataclass
class EnergySchedConfig:
    beta: float = 1.0
    w_ttft: float = 10.0
    w_tpot: float = 1.0
    eta_ttft: float = 0.0
    eta_tpot: float = 0.0
    eta_ms: float = 1e9
    Lmax: int = 0
    max_batch_size: int = 0
    default_w_n: float = 1.0
    default_ttft_ms: float = 4000.0
    default_tpot_ms: float = 200.0
    freq_candidates: Optional[List[int]] = None
    freq_stride: int = 1
    solution_mode: int = 1
    chunked_prefill: bool = False
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
    wait_ms: float
    deadline_ms: float
    w_n: float
    is_waiting: bool = False
    kv_blocks_needed: int = 0
    kv_blocks_incremental: int = 0


def baseline_reward(r: ReqView, cfg: EnergySchedConfig) -> float:
    return r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)


class Alt1HeuristicSolver:
    """Heuristic solver (H2, H3, and H4 modes).

    Step 1: priority q_n = r_n * min(exp(-s_n), CAP) / l_n
    Step 2: density-greedy fill until L_max or B_max binds
    Step 3: mode-dependent prefix/frequency selection, pick argmax utility
    """

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
        self.freq_candidates = sorted(freq_candidates)

    def solve(
        self,
        reqs: List[ReqView],
        Lmax: int,
        Bmax: int,
        debug_iter: int = -1,
        waiting_capacity: Optional[int] = None,
    ) -> Tuple[float, list, float]:
        if self.cfg.solution_mode == 3:
            return self._solve_h4(reqs, Lmax, Bmax, debug_iter, waiting_capacity)
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

        t_c_s = lat.t_c / 1000.0

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

        wp_contrib = np.where(
            is_pf, lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q, 0.0
        )
        wd_contrib = np.where(is_pf, 0.0, lat.a_d * l_kv + lat.b_d)

        cap = float(self.EXP_CAP)
        urgency = np.minimum(np.exp(-s_n_s), cap)
        ell_safe = np.maximum(tok_arr.astype(np.float64), 1.0)
        q_n = r_n_vec * urgency / ell_safe

        order = np.argsort(-q_n, kind="stable")
        B_eff = int(Bmax) if Bmax > 0 else N
        used_tok = 0
        picked_local: List[int] = []
        chunked = self.cfg.chunked_prefill

        for idx in order:
            if len(picked_local) >= B_eff:
                break
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

        if not picked_local:
            return float(default_f), [], 0.0

        K = len(picked_local)
        picked_idx = np.asarray(picked_local, dtype=np.int64)
        is_pf_picked = is_pf[picked_idx]

        dp = wp_contrib[picked_idx].astype(np.float64)
        dd = wd_contrib[picked_idx].astype(np.float64)
        cum_dp = np.cumsum(dp)
        cum_dd = np.cumsum(dd)
        cum_has_pf = (np.cumsum(is_pf_picked.astype(np.int64)) > 0).astype(np.float64)
        cum_has_dc = (np.cumsum((~is_pf_picked).astype(np.int64)) > 0).astype(np.float64)

        num_p_vec = cum_dp + lat.w_pf * cum_has_pf
        num_d_vec = cum_dd + lat.w_dec * cum_has_dc

        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates
        if self.freq_candidates and self.freq_candidates[-1] not in freqs:
            freqs = freqs + [self.freq_candidates[-1]]
        f_arr = np.asarray(freqs, dtype=np.float64)
        f_alpha_arr = f_arr ** lat.alpha
        F = f_arr.size

        ET_s_mat = (
            num_p_vec[None, :] / f_arr[:, None]
            + num_d_vec[None, :] / f_alpha_arr[:, None]
            + lat.t_c
        ) / 1000.0

        s_picked = s_n_s[picked_idx]
        r_picked = r_n_vec[picked_idx]

        overshoot = (
            ET_s_mat[:, :, None] - s_picked[None, None, :]
        )
        np.maximum(overshoot, 0.0, out=overshoot)
        u_mat = r_picked[None, None, :] * np.exp(-overshoot)

        tri = np.tri(K, K, dtype=np.float64)
        sum_u_per_jf = (u_mat * tri[None, :, :]).sum(axis=2)

        P_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )
        J_mat = sum_u_per_jf - beta * P_arr[:, None] * ET_s_mat

        flat_idx = int(np.argmax(J_mat))
        best_fi, best_jidx = np.unravel_index(flat_idx, J_mat.shape)
        best_j_size = int(best_jidx) + 1
        best_f = float(f_arr[best_fi])
        best_et_s = float(ET_s_mat[best_fi, best_jidx])
        chosen_local = picked_local[:best_j_size]

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in chosen_local if reqs[i].is_prefill)
            n_d_ch = best_j_size - n_p_ch
            print(
                f"[dbg-h2] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"greedy=B̂[{K}] chosen=B_{best_j_size}(p={n_p_ch}d={n_d_ch}) "
                f"f={best_f:.0f} ET={best_et_s*1000.0:.2f}ms "
                f"B_max={B_eff} CAP={self.EXP_CAP}",
                file=sys.stderr, flush=True)

        return float(best_f), [reqs[i] for i in chosen_local], best_et_s * 1000.0

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

        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates
        if self.freq_candidates and self.freq_candidates[-1] not in freqs:
            freqs = freqs + [self.freq_candidates[-1]]
        #freqs = [1410]
        f_arr = np.asarray(freqs, dtype=np.float64)
        F = f_arr.size
        f_alpha_arr = f_arr ** lat.alpha
        P_f_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )
        RU = r_n_vec * urgency
        t_nf_all = (
            wp_contrib[None, :] / f_arr[:, None]
            + wd_contrib[None, :] / f_alpha_arr[:, None]
        ) / 1000.0
        q_all = (
            RU[None, :] - beta * P_f_arr[:, None] * t_nf_all
        ) / ell_safe[None, :]
        orders_all = np.argsort(-q_all, axis=1, kind="stable")

        exp_s_all = np.exp(s_n_s)

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
            order_fi = orders_all[fi]

            chunked_override: dict = {}
            tok_sorted = tok_arr[order_fi]
            B_try = min(B_eff, N)
            cum_tok_b = np.cumsum(tok_sorted[:B_try])
            if cum_tok_b[B_try - 1] <= Lmax:
                picked_local = order_fi[:B_try].tolist()
            else:
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

            exp_s = exp_s_all[picked_idx]
            exp_neg_ET = np.exp(-ET_s)
            raw = exp_neg_ET[:, None] * exp_s[None, :]
            u_mat = r_picked[None, :] * np.minimum(raw, 1.0)
            tri = np.tri(K, K, dtype=np.float64)
            sum_u = (u_mat * tri).sum(axis=1)

            J_vec = sum_u - beta * P_f * ET_s
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
                f"f={global_best_f:.0f} "
                f"ET={global_best_et_s*1000.0:.2f}ms "
                f"B_max={B_eff} CAP={self.EXP_CAP}",
                file=sys.stderr, flush=True)

        return float(global_best_f), [reqs[i] for i in global_best_chosen], global_best_et_s * 1000.0

    def _solve_h4(
        self,
        reqs: List[ReqView],
        Lmax: int,
        Bmax: int,
        debug_iter: int = -1,
        waiting_capacity: Optional[int] = None,
    ) -> Tuple[float, list, float]:
        default_f = self.freq_candidates[-1] if self.freq_candidates else 1410
        if not reqs:
            return float(default_f), [], 0.0

        N = len(reqs)
        cfg = self.cfg
        lat = self.latency
        beta = cfg.beta
        cap = float(self.EXP_CAP)

        is_pf = np.fromiter((r.is_prefill for r in reqs), dtype=bool, count=N)
        is_waiting = np.fromiter((r.is_waiting for r in reqs), dtype=bool, count=N)
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
        deadline_safe_s = np.maximum(deadline_s, 1e-6)
        normalized_slack = s_n_s / deadline_safe_s
        urgency = np.minimum(np.exp(-normalized_slack), cap)
        ell_safe = np.maximum(tok_arr.astype(np.float64), 1.0)

        wp_contrib = np.where(
            is_pf, lat.a_p * l_q * l_q + lat.b_p * l_q * l_kv + lat.c_p * l_q, 0.0
        )
        wd_contrib = np.where(is_pf, 0.0, lat.a_d * l_kv + lat.b_d)

        B_eff = int(Bmax) if Bmax > 0 else N
        waiting_cap = N if waiting_capacity is None else max(0, int(waiting_capacity))

        stride = cfg.freq_stride
        freqs = self.freq_candidates[::stride]
        if not freqs:
            freqs = self.freq_candidates
        if self.freq_candidates and self.freq_candidates[-1] not in freqs:
            freqs = freqs + [self.freq_candidates[-1]]
        #freqs = [1410]
        f_arr = np.asarray(freqs, dtype=np.float64)
        F = f_arr.size
        f_alpha_arr = f_arr ** lat.alpha
        P_f_arr = np.array(
            [self.power.power_watts(float(f)) for f in freqs], dtype=np.float64
        )

        RU = r_n_vec * urgency
        t_nf_all = (
            wp_contrib[None, :] / f_arr[:, None]
            + wd_contrib[None, :] / f_alpha_arr[:, None]
        ) / 1000.0
        q_all = (
            RU[None, :] - beta * P_f_arr[:, None] * t_nf_all
        ) / ell_safe[None, :]
        orders_all = np.argsort(-q_all, axis=1, kind="stable")

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
            order_fi = orders_all[fi]

            picked_local: List[int] = []
            chunked_override: dict = {}
            used_tok = 0
            used_waiting = 0
            q_sorted = q_all[fi, order_fi]

            for pos, orig_idx in enumerate(order_fi.tolist()):
                if len(picked_local) >= B_eff:
                    break
                if float(q_sorted[pos]) <= 0.0:
                    break
                if bool(is_waiting[orig_idx]) and used_waiting >= waiting_cap:
                    continue

                tok_n = int(tok_arr[orig_idx])
                if used_tok + tok_n > Lmax:
                    if chunked and bool(is_pf[orig_idx]):
                        tok_n = Lmax - used_tok
                        if tok_n <= 0:
                            break
                        actual_lq = float(tok_n)
                        actual_lkv = float(l_kv[orig_idx])
                        new_wp = (lat.a_p * actual_lq * actual_lq
                                  + lat.b_p * actual_lq * actual_lkv
                                  + lat.c_p * actual_lq)
                        chunked_override[len(picked_local)] = (new_wp, tok_n)
                    else:
                        break

                picked_local.append(int(orig_idx))
                if bool(is_waiting[orig_idx]):
                    used_waiting += 1
                used_tok += tok_n

            if not picked_local:
                continue

            picked_idx = np.asarray(picked_local, dtype=np.int64)
            is_pf_picked = is_pf[picked_idx]

            dp = wp_contrib[picked_idx].astype(np.float64)
            if chunked_override:
                for pos, (new_wp, _) in chunked_override.items():
                    dp[pos] = new_wp
            dd = wd_contrib[picked_idx].astype(np.float64)

            has_pf = bool(is_pf_picked.any())
            has_dc = bool((~is_pf_picked).any())
            num_p = float(dp.sum()) + (lat.w_pf if has_pf else 0.0)
            num_d = float(dd.sum()) + (lat.w_dec if has_dc else 0.0)
            ET_s = (num_p / f_float + num_d / f_alpha + lat.t_c) / 1000.0

            overshoot = np.maximum(ET_s - s_n_s[picked_idx], 0.0)
            #overshoot = np.array(s_n_s[picked_idx])
            overshoot = overshoot / deadline_safe_s[picked_idx]
            utility = float((r_n_vec[picked_idx] * np.exp(-overshoot)).sum())
            J_f = utility - beta * P_f * ET_s

            if J_f > global_best_J:
                global_best_J = float(J_f)
                global_best_f = f_float
                global_best_chosen = picked_local
                global_best_et_s = float(ET_s)
                global_best_chunked = dict(chunked_override)

        if not global_best_chosen:
            return float(default_f), [], 0.0

        for pos, (_, actual_lq) in global_best_chunked.items():
            if pos < len(global_best_chosen):
                reqs[global_best_chosen[pos]].l_q = actual_lq

        if debug_iter >= 0 and debug_iter % 10 == 0:
            import sys
            n_p = int(is_pf.sum())
            n_d = N - n_p
            n_p_ch = sum(1 for i in global_best_chosen if reqs[i].is_prefill)
            n_d_ch = len(global_best_chosen) - n_p_ch
            print(
                f"[dbg-h4] iter={debug_iter} all={N}(p={n_p}d={n_d}) "
                f"chosen=B_{len(global_best_chosen)}(p={n_p_ch}d={n_d_ch}) "
                f"f={global_best_f:.0f} J*={global_best_J:.3f} "
                f"ET={global_best_et_s*1000.0:.2f}ms "
                f"B_max={B_eff} CAP={self.EXP_CAP}",
                file=sys.stderr, flush=True)

        return float(global_best_f), [reqs[i] for i in global_best_chosen], global_best_et_s * 1000.0


FrequencyFirstSolver = Alt1HeuristicSolver


def _open_iter_log(path: Optional[str]):
    if path is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "a")
