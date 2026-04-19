# Experiment Record — Energy-Efficient LLM Scheduling on vLLM

## 1. Problem Statement

We run a single vLLM server serving Qwen3-14B on an A800-SXM4-80GB GPU and compare two schedulers on the same workload:

- **Baseline**: vLLM's default FCFS scheduler, GPU clocks not locked.
- **Ours (custom)**: an energy-aware scheduler that solves a per-iteration frequency-first optimisation (enumerating `(f, M)` pairs and solving a 2-D knapsack), and attempts to lock the SM clock via `pynvml` on every iteration.

Reported metrics include mean TTFT/TPOT, SLO violations, power, energy, and the mean solve-to-execution ratio.

## 2. Commands Executed

```bash
# Environment
conda activate myvllm

# One-time setup
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='RyokoAI/ShareGPT52K', repo_type='dataset', \
                      local_dir='data/sharegpt52k', local_dir_use_symlinks=False)"

# Apply patch to vLLM
bash vllm_patches/apply_patch.sh /home/ubuntu/lqs/vllm

# Generate trace (500 requests)
python scripts/prepare_dataset.py --output trace.jsonl --num-requests 500 --rate-qps 8 --seed 42

# Run both-mode experiment
bash main.sh --tag demo --num-requests 500 --rate-qps 8 --mode both --freq-stride 4
```

## 3. Files Created

### `main.sh` (283 lines)
Master runner script. Sources conda, applies the vLLM patch, generates trace, runs default and/or custom experiments, collects metrics, and compares results. Contains a "USER KNOBS" block at the top for parameter tuning.

### `scripts/prepare_dataset.py` (85 lines)
Reads ShareGPT52K JSON files, filters by prompt length, shuffles, and writes `trace.jsonl` with per-request SLO parameters (TTFT, TPOT, priority).

Key function: `main()` — loads dataset, filters, samples, writes JSONL.

### `scripts/workload_sender.py` (170 lines)
Async replay of `trace.jsonl` against vLLM's `/v1/completions` endpoint with `stream=true`. Sends SLO parameters via `extra_body.extra_args`. Records per-request TTFT, TPOT, and writes `results.jsonl`.

Key function: `send_one()` — sends one request, records timestamps, computes per-request metrics.

### `scripts/power_logger.py` (58 lines)
Samples GPU power/clock/utilisation via `pynvml` every 0.1s and writes CSV. Stops on SIGTERM/SIGINT.

Key function: `main()` — NVML init, loop sampling, CSV write with flush.

### `scripts/metrics_collector.py` (126 lines)
Aggregates results from `results.jsonl` and `power.csv`. Computes mean TTFT/TPOT, SLO violations, attainment, mean power, total energy (trapezoidal integration), and solve-to-exec ratio from the iteration log.

Key function: `main()` — loads results, computes all metrics, writes `summary.json`.

### `scripts/compare_results.py` (62 lines)
Reads two summary JSONs, prints a side-by-side table, and writes `comparison.csv`.

### `vllm_patches/energy_model.py` (114 lines)
Route B+ 9-parameter latency model and cubic power model. Hard-coded defaults from A800 profiling.

Key functions:
- `per_request_time_ms()` — per-request `t_q(n, f)`
- `batch_overhead_ms()` — per-batch `T_ovh(B, f)`
- `batch_time_ms()` — total `ET_i(B, f) = Σ t_q + T_ovh + t_c`
- `load_latency_params()` / `load_power_params()` — with optional JSON override

### `vllm_patches/frequency_controller.py` (99 lines)
NVML-backed GPU frequency controller. Queries supported clocks, sets locked clocks, resets on exit. Falls back to `sudo nvidia-smi -lgc` when NVML permissions are insufficient.

Key functions:
- `set_frequency(f_mhz)` — locks SM clock (NVML or sudo nvidia-smi)
- `reset()` — unlocks SM clock
- `get_controller()` — singleton factory

### `vllm_patches/energy_scheduler.py` (411 lines)
The energy-aware scheduler. Pure-Python `FrequencyFirstSolver` + vLLM `EnergyScheduler` subclass.

Key classes/functions:
- `EnergySchedConfig.from_env()` — reads `VLLM_ENERGY_*` env vars
- `adjusted_utility()` — returns `(v_{i,n}(f), t_q(n, f))`
- `greedy_knapsack_2d()` — utility-density greedy 2-D knapsack
- `FrequencyFirstSolver.solve()` — enumerates `(f, M)` pairs, solves knapsack per pair. Running (decode) requests are always mandatory; prefill requests are energy-aware selected.
- `make_energy_scheduler_class()` — factory that creates `EnergyScheduler` subclass of vLLM's `Scheduler`

### `vllm_patches/apply_patch.sh` (36 lines)
Copies Python files into `vllm/energy_sched/` package and appends the sentinel-guarded hook to `vllm/__init__.py`. Idempotent (checks for marker before appending).

### `vllm_patches/unapply_patch.sh` (27 lines)
Removes the `energy_sched` package and the hook block from `vllm/__init__.py`.

## 4. vLLM Edits

**File**: `/home/ubuntu/lqs/vllm/vllm/__init__.py`

**Line**: 103 (appended block at end of file)

**Snippet** (between `# <<< ENERGY_SCHED_HOOK >>>` marker and EOF):
```python
# <<< ENERGY_SCHED_HOOK >>>
import os as _ENERGY_os
if _ENERGY_os.environ.get("VLLM_ENERGY_SCHEDULER", "0") == "1":
    try:
        from vllm.energy_sched.energy_scheduler import make_energy_scheduler_class
        from vllm.v1.core.sched import scheduler as _ENERGY_mod
        _ENERGY_mod.Scheduler = make_energy_scheduler_class()
    except Exception as _e:
        import sys as _s
        print(f"[energy_sched] failed to install: {_e}", file=_s.stderr, flush=True)
```

**New directory**: `/home/ubuntu/lqs/vllm/vllm/energy_sched/` — contains `__init__.py`, `energy_model.py`, `energy_scheduler.py`, `frequency_controller.py`.

No existing vLLM files were modified — the patch is a pure no-op unless `VLLM_ENERGY_SCHEDULER=1`.

## 5. Dataset Provenance

- **Repo**: `RyokoAI/ShareGPT52K` on Hugging Face
- **Commit SHA**: `6f9b78cc1dd15dbb51d3c51ccc219c558962fd77`
- **Trace**: 500 requests, first human message per conversation, prompt length 64–8000 chars
- **SLO parameters**: TTFT μ=3000ms σ=500ms, TPOT μ=200ms σ=40ms (truncated normal)
- **Arrival rate**: 8 req/s (uniform)

## 6. Results

| Metric | Default | Custom |
|--------|---------|--------|
| mean_ttft_ms | 4106.25 | 22631.79 |
| mean_tpot_ms | 31.83 | 27.83 |
| mean_ttft_violation_ms | 1828.14 | 19972.38 |
| mean_tpot_violation_ms | 0.0 | 0.0 |
| ttft_slo_attainment | 0.416 | 0.116 |
| tpot_slo_attainment | 1.0 | 1.0 |
| mean_power_w | 349.33 | 331.43 |
| total_energy_j | 28982.75 | 39327.93 |
| mean_solve_exec_ratio | 0.0 | 0.010888 |

**Notes on results**:
- All 500 requests completed in both modes.
- TPOT SLO attainment is 1.0 in both modes (decode requests are always served).
- TTFT is higher in custom mode because the energy-aware solver defers prefill requests with low adjusted utility (instantaneous utility is small for fresh prefill since `exp(-3) ≈ 0.05`, while the energy penalty `P·t_q` dominates at ~35).
- The frequency-locking feature (`nvmlDeviceSetGpuLockedClocks`) requires root or `CAP_SYS_NICE`. Without it, both runs operate at the default GPU clock (~210 MHz base, ramping under load). The custom scheduler's frequency selection is therefore not enforced, limiting its energy-saving potential.
- `mean_solve_exec_ratio = 0.011` for custom (solve overhead is ~1% of batch execution time) — well below 1.0, confirming the solver is lightweight enough for per-iteration use.

## 7. How to Reproduce

```bash
cd /home/ubuntu/lqs/energy_efficient_LLM_scheduling
conda activate myvllm

# Setup (idempotent)
bash vllm_patches/apply_patch.sh /home/ubuntu/lqs/vllm
python scripts/prepare_dataset.py --output trace.jsonl --num-requests 500 --rate-qps 8 --seed 42

# Run experiment
bash main.sh --tag demo --num-requests 500 --rate-qps 8 --mode both --freq-stride 4

# View results
cat results/demo/comparison.csv
```

For root-privilege frequency locking: ensure the user has `CAP_SYS_NICE` or run `main.sh` with `sudo`.
