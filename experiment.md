# Experiment Record — Energy-Efficient LLM Scheduling on vLLM

## 1. Problem Statement

We run a single vLLM server serving Qwen3-14B on an A800-SXM4-80GB GPU and compare two schedulers on the same workload:

- **Baseline**: vLLM's default FCFS scheduler, GPU clocks not locked.
- **Ours (custom)**: an energy-aware scheduler based on the **Heuristic 4 (H4)** formulation — a two-step algorithm (frequency-dependent priority scoring with normalized slack → greedy fill with q_n≤0 cutoff) with online adaptive weight updates. The scheduler selects both the GPU SM frequency and batch composition per iteration, locking the SM clock via `pynvml`.

Current default configuration: `SOLUTION_MODE=3` (H4), `IS_COOLDOWN=2` (priority decay on preemption), `IS_CHUNKED_PREFILL=1` (chunked prefill enabled).

Reported metrics include mean TTFT/TPOT, SLO violations (absolute and normalized), SLO attainment, power, energy, and the mean solve-to-execution ratio.

## 2. Commands Executed

```bash
# Environment
conda activate myvllm

# Run experiment (dataset auto-download + patch + trace + experiment)
bash main.sh
```

`main.sh` handles everything end-to-end: applying the vLLM patch, verifying/downloading the dataset, generating the trace, running experiments, collecting metrics, and comparing results.

## 3. Files Created — Full Code Review

### 3.1 `main.sh` (305 lines) — Master experiment orchestrator

**Purpose**: Controls the full experiment lifecycle — applies the vLLM patch, generates the workload trace, launches the vLLM server, replays the workload, logs power, collects metrics, and compares results.

**L1–112: USER KNOBS block** — All tunable parameters are declared as Bash variables at the top of the file:

| Variable | Default | Meaning |
|---|---|---|
| `TAG` | `"beta_0.4"` | Output directory name under `results/` |
| `MODE` | `"both"` | Which scheduler: `"default"`, `"custom"`, or `"both"` |
| `VLLM_DIR` | `/home/ubuntu/lqs/vllm` | Path to local vLLM source tree |
| `MODEL_DIR` | `/home/ubuntu/lqs/LLM_model` | Path to model weights (Qwen3-14B) |
| `MODEL_NAME` | `"default"` | Model name served to clients |
| `PORT` | `8000` | HTTP port for vLLM API server |
| `GPU_INDEX` | `0` | Which GPU to use (CUDA device index) |
| `MAX_MODEL_LEN` | `8192` | Max sequence length (input + output tokens) |
| `DEFAULT_MAX_NUM_SEQS` | `256` | Active cap for default scheduler |
| `CUSTOM_MAX_NUM_SEQS` | `400` | Active cap for custom scheduler (larger to give solver more candidates) |
| `GPU_MEM_UTIL` | `0.95` | Fraction of GPU memory for KV cache |
| `TRACE_SEED` | `42` | Random seed for trace generation |
| `BETA` | `1.0` | Energy-utility trade-off (larger = more energy-saving) |
| `W_TTFT` | `1000` | Initial weight for TTFT in priority calculation (mutable — drifts online) |
| `W_TPOT` | `100` | Initial weight for TPOT in priority calculation (mutable — drifts online) |
| `VLLM_MAX_BATCHED_TOKENS` | `8192` | Passed to vLLM `--max-num-batched-tokens`; must be ≥ MAX_MODEL_LEN |
| `SOLVER_LMAX` | `8192` | Solver-side max tokens per batch in greedy fill |
| `FREQ_STRIDE` | `3` | Stride for frequency candidate subsampling |
| `MAX_BATCH_SIZE` | `256` | Max requests per iteration (batch cap) |
| `IS_COOLDOWN` | `2` | Preemption handling: 1=TTFT-SLO cooldown, 2=effective `w_n` decay |
| `DECAY_PARAMETER` | `10000` | Only for `IS_COOLDOWN=2`; each preemption divides the request multiplier by this value |
| `SOLUTION_MODE` | `3` | Solver heuristic (1=H2, 2=H3, 3=H4, 4=H5) |
| `IS_CHUNKED_PREFILL` | `1` | 0=non-chunked prefill, 1=chunked prefill |
| `POWER_INTERVAL_S` | `0.05` | GPU power sampling interval (seconds) |

**L114**: Captures the script directory so all paths are absolute regardless of CWD.

**L117–127**: Conda activation. Tries four possible `conda.sh` locations (miniconda3, anaconda3, /opt/conda), sources the first one found, then activates the `myvllm` environment.

**L132–133**: Calls `apply_patch.sh` to copy the energy scheduler Python files into the vLLM source tree and apply the scheduler patch.

**L136–141**: Conditional trace generation. If `trace.jsonl` already exists, it is reused. Delete the file to force regeneration.

**L145–154**: `reset_gpu_clocks()` helper — uses `FrequencyController` to reset GPU clocks, falls back to `nvidia-smi -rgc / -rmc`.

**L157–287: `run_experiment()` function** — The core experiment runner:

- **L158–163**: Maps `"default"`/`"custom"` label to the output file suffix.
- **L169–190**: Builds the server environment variable array. For baseline, `VLLM_ENERGY_SCHEDULER=0`. For custom mode, sets `VLLM_ENERGY_SCHEDULER=1` plus all hyperparameters (`VLLM_ENERGY_BETA`, `VLLM_ENERGY_W_TTFT`, `VLLM_ENERGY_W_TPOT`, `VLLM_ENERGY_LMAX`, `VLLM_ENERGY_MAX_BATCH_SIZE`, `VLLM_ENERGY_PREEMPT_MODE`, `VLLM_ENERGY_PREEMPT_DECAY_PARAMETER`, `VLLM_ENERGY_PREEMPT_MIN_MULTIPLIER`, `VLLM_ENERGY_FREQ_STRIDE`, `VLLM_ENERGY_SOLUTION_MODE`, `VLLM_ENERGY_GPU_INDEX`, `VLLM_ENERGY_ITER_LOG`, `VLLM_ENERGY_CHUNKED_PREFILL`).
- **L197–224**: Launches the vLLM server as a background process. Key flags:
  - `--enforce-eager`: Disables CUDA graphs (needed because frequency changes invalidate graph caches)
  - `--no-async-scheduling`: Disables async scheduling so the scheduler sees all running/waiting requests at each iteration
  - `--enable-chunked-prefill` or `--no-enable-chunked-prefill`: Controlled by `IS_CHUNKED_PREFILL` knob
  - `--max-num-batched-tokens`: Set from `VLLM_MAX_BATCHED_TOKENS` if > 0
  - `--no-enable-prefix-caching`: Disables prefix caching (simplifies KV cache accounting)
  - `--enable-logging-iteration-details`: Enables detailed per-iteration logging
- **L227–240**: Health check loop. Polls `http://localhost:PORT/health` every 2 seconds for up to 240 seconds.
- **L243–248**: Starts `power_logger.py` as a background process.
- **L251–256**: Runs `workload_sender.py` synchronously — blocks until all requests are done.
- **L259–266**: Stops power logger and server via `kill`.
- **L269–271**: Resets GPU clocks to default after custom mode.
- **L274–285**: Runs `metrics_collector.py` to aggregate results into `summary_${label}.json`.

**L290–296**: Sequential experiment execution. If `MODE` is `"default"` or `"both"`, runs baseline. If `"custom"` or `"both"`, runs custom.

**L299–305**: Runs `compare_results.py` to produce a side-by-side comparison table and CSV.

---

### 3.2 `scripts/prepare_dataset.py` (162 lines) — Trace generation with discrete SLO classes

**Purpose**: Ensures the ShareGPT52K dataset is available (auto-downloading if needed), filters and samples prompts, tokenizes reference outputs to determine `max_tokens`, and writes `trace.jsonl` — one JSON record per line representing a single request with its arrival time, prompt, and SLO parameters.

**L17–55: USER KNOBS block**:

| Constant | Default | Meaning |
|---|---|---|
| `OUTPUT` | `"trace.jsonl"` | Output file path |
| `NUM_REQUESTS` | `1000` | Number of requests to sample |
| `RATE_QPS` | `3` | Arrival rate — request i arrives at `i / RATE_QPS` seconds |
| `SLO_CLASSES` | (see below) | Discrete SLO tiers with weighted random assignment |
| `MIN_PROMPT_CHARS` | `512` | Minimum prompt length (characters) |
| `MAX_PROMPT_CHARS` | `6000` | Maximum prompt length (characters) |
| `SEED` | `42` | Random seed for reproducibility |
| `DATASET_DIR` | `"data/sharegpt52k"` | Local path to ShareGPT dataset |
| `TOKENIZER_DIR` | `/home/ubuntu/lqs/LLM_model` | Tokenizer used to count reference output tokens |
| `REPO_ID` | `"RyokoAI/ShareGPT52K"` | Hugging Face repository ID for auto-download |

**SLO classes** (replacing the old truncated normal sampling):

| Class | TTFT SLO (ms) | TPOT SLO (ms) | Weight |
|---|---|---|---|
| `strict` | 600 | 80 | 0.30 |
| `normal` | 1000 | 100 | 0.50 |
| `relaxed` | 1500 | 150 | 0.20 |

Each request is assigned one SLO class via `random.choices()` with the above weights. This replaces the previous truncated normal sampling (μ=4000ms TTFT, μ=100ms TPOT) with tighter, more realistic SLO targets.

**L61–81: `_ensure_dataset()`** — Automatic dataset verification. Same logic as before: downloads via `huggingface_hub.snapshot_download()` if missing; re-downloads if Git LFS pointers are detected.

**L102–126: Dataset loading**:
- Iterates all `.json` files in `DATASET_DIR`.
- For each conversation, extracts the first human/user message.
- Filters by prompt length (512–6000 characters).
- Finds the reference assistant output and tokenizes it via `AutoTokenizer` to determine `max_tokens` — this replaces the old uniform random sampling from `[64, 1024]`.

**L137–156: Trace writing**:
- Shuffles all candidate prompts, takes the first `NUM_REQUESTS`.
- For each request, writes a JSON record with:
  - `id`: unique identifier like `"req_000001"`
  - `arrival_s`: `i / RATE_QPS` — uniform arrival times
  - `prompt`: the actual text content
  - `max_tokens`: actual tokenized reference output length (not randomly sampled)
  - `slo_class`: the assigned SLO class name (`"strict"`, `"normal"`, or `"relaxed"`)
  - `ttft_ms`: SLO target from the assigned class
  - `tpot_ms`: SLO target from the assigned class
  - `w_n`: priority weight, default 1.0

---

### 3.3 `scripts/workload_sender.py` (179 lines) — Async workload replay

**Purpose**: Reads `trace.jsonl` and asynchronously sends each request to the vLLM `/v1/completions` endpoint with `stream=true`, measuring per-request TTFT and TPOT.

**L136**: Custom TCP connector — `aiohttp.TCPConnector(limit=1000, limit_per_host=1000)` raises the connection pool ceiling to avoid HTTP-layer queuing under high concurrency.

**L19–34: `ResultRecord` dataclass**: Holds per-request metadata and results:
- `id`, `prompt`, `max_tokens`, `ttft_slo_ms`, `tpot_slo_ms`, `w_n`, `arrival_s`
- `send_time`: wall-clock epoch (seconds) recorded just before the HTTP POST
- `complete_time`: wall-clock epoch when the request finishes
- `status`, `ttft_ms`, `tpot_ms`, `num_output_tokens`, `error`

**L37–116: `send_one()`**: Sends a single request and measures timing:
- Builds the HTTP POST payload; passes TTFT/TPOT/w_n/`send_time` to the server via `vllm_xargs`.
- Parses the SSE stream, counting tokens and recording inter-chunk gaps for TPOT.
- TTFT is measured as `(first_chunk_time - send_time) * 1000.0`.

**Key design — `send_time`**: The workload sender records `send_time` (wall-clock epoch) *before* the HTTP POST and passes it to the server via `vllm_xargs.send_time`. The energy scheduler's `_energy_get_arrival()` uses `send_time` as the authoritative arrival time, falling back to `req.arrival_time` only if absent. This captures the full end-to-end latency including HTTP and engine-queue delays, avoiding underestimation from vLLM's internal `arrival_time`.

**L119–176: `main()`**: Orchestrates the workload replay, dispatching requests respecting arrival times via `asyncio.sleep`, then writes `results.jsonl`.

---

### 3.4 `scripts/power_logger.py` (57 lines) — GPU power sampling

**Purpose**: Continuously samples GPU power draw, SM clock frequency, and GPU utilization via `pynvml`, writing a CSV row per sample.

- **Signal handler**: Sets `_stop` flag on SIGTERM/SIGINT for clean shutdown.
- **CSV columns**: `timestamp_s, power_w, sm_clock_mhz, utilization_pct`
- **Sampling loop**: Calls `nvmlDeviceGetPowerUsage()` (mW→W), `nvmlDeviceGetClockInfo()`, `nvmlDeviceGetUtilizationRates()`. Writes with `flush=True`; sleeps for `POWER_INTERVAL_S` (0.05s by default).

---

### 3.5 `scripts/metrics_collector.py` (200 lines) — Metrics aggregation

**Purpose**: Reads `results.jsonl` and `power.csv` from a completed experiment, computes summary statistics, and writes `summary.json`.

- **`trapz()`**: Trapezoidal integration (time × power → energy in joules).
- **`interpolate_power()`**: Linear interpolation of the power trace at arbitrary timestamps.
- **`windowed_energy()`**: Computes energy and mean power over `[first_send_time, last_complete_time]` — the active period only.
- **`solve_exec_ratio()`**: Reads iteration log, returns mean of `solve_ms / exec_ms`.
- **Summary metrics**: mean TTFT/TPOT, absolute and **normalized** SLO violations (`max(0, obs − slo) / slo`), SLO attainment, windowed energy, and solve-exec ratio.

---

### 3.6 `scripts/compare_results.py` (60 lines) — Result comparison

**Purpose**: Reads two `summary.json` files (default and custom), prints a side-by-side table, and writes `comparison.csv`. Comparison metrics: `mean_ttft_violation_ms`, `mean_tpot_violation_ms`, `mean_normalized_ttft_violation`, `mean_normalized_tpot_violation`, `mean_power_w`, `total_energy_j`, `mean_solve_exec_ratio`.

---

### 3.7 `vllm_patches/energy_model.py` (134 lines) — Latency and power models

**Purpose**: Provides the mathematical models for per-request latency and GPU power as functions of frequency. These are used by the solver to predict execution time and energy consumption.

**`LatencyParams` dataclass** — 9 coefficients fitted to A800-SXM4-80GB profiling data (current fit):

| Parameter | Value | Meaning |
|---|---|---|
| `a_p` | `0.0` | Prefill: quadratic term (prompt-length²) — currently zero |
| `b_p` | `6.0e-3` | Prefill: cross-term (prompt × KV context length) |
| `c_p` | `145.55` | Prefill: linear term (prompt-length) |
| `w_pf` | `5000.2` | Batch overhead weight for prefill |
| `w_dec` | `15000` | Batch overhead weight for decode |
| `a_d` | `0.1675` | Decode: linear coefficient on KV length |
| `b_d` | `102.64` | Decode: constant term |
| `alpha` | `0.9881` | Frequency scaling exponent for decode |
| `t_c` | `6.582` | Constant communication overhead (ms) |

**`PowerParams` dataclass**: Cubic power model `P(f) = k3·f³ + k2·f² + k1·f + k0`.

**`per_request_time_ms()`** — Per-request latency contribution:
- **Prefill**: `t_q = (a_p · l_q² + b_p · l_q · l_kv + c_p · l_q) / f`
- **Decode**: `t_q = (a_d · l_kv + b_d) / f^α`

**`batch_overhead_ms()`** — Mode-dependent batch overhead:
- `T_ovh = I_p · w_pf/f + I_d · w_dec/f^α`

**`batch_time_ms()`** — Total iteration time:
- `ET_i(B, f) = Σ_{n∈B} t_q(n, f) + T_ovh(B, f) + t_c`

---

### 3.8 `vllm_patches/frequency_controller.py` (152 lines) — GPU frequency control

**Purpose**: Provides an abstraction for setting and resetting GPU SM clock frequency and memory clock frequency. Uses `pynvml` as the primary mechanism, falls back to `sudo nvidia-smi` when permissions are insufficient.

- `__init__`: Initializes NVML, queries supported SM clocks, queries memory clocks, attempts to lock memory clock to 1593 MHz (profiling baseline — no-op on A800).
- `supported_clocks()`: Returns sorted list of supported SM frequencies (81 values on A800: 210–1410 MHz, 15 MHz steps).
- `set_frequency(f_mhz)`: Finds closest supported frequency, skips if already set, tries pynvml then `sudo nvidia-smi -lgc`.
- `reset()`: Unlocks SM and memory clocks.
- `get_controller()`: LRU-cached singleton factory.

---

### 3.9 Energy Scheduler Architecture

The energy scheduler is split into two layers:

1. **`vllm_patches/solver.py`** — Pure algorithm (no vLLM imports, only numpy + stdlib)
2. **`vllm/v1/core/sched/scheduler.py`** — vLLM integration (applied via git patch)

#### 3.9.1 `vllm_patches/solver.py` (848 lines) — Algorithm Layer

Contains `EnergySchedConfig`, `ReqView`, `Alt1HeuristicSolver` (H2 + H3 + H4 + H5), `baseline_reward()`, `_open_iter_log()`.

##### `ReqView` dataclass

| Field | Type | Meaning |
|---|---|---|
| `handle` | Any | vLLM request object |
| `is_prefill` | bool | True if compute type is prefill (context/prompt processing) |
| `l_q` | int | Per-iter token cost (prompt tokens for prefill, 1 for decode) |
| `l_kv` | int | KV cache length (computed tokens) |
| `wait_ms` | float | Time since arrival/last output (ms) |
| `deadline_ms` | float | SLO deadline (TTFT for prefill, TPOT for decode, in ms) |
| `w_n` | float | Per-request priority weight |
| `is_waiting` | bool | True if request is in the waiting queue (used by H4/H5 for admission cap) |
| `kv_blocks_needed` | int | Full KV size in blocks |
| `kv_blocks_incremental` | int | New blocks needed this iteration |
| `slo_is_ttft` | Optional[bool] | SLO type override. `None` = follow `is_prefill`; `False` = preempted decode doing context recomputation (compute=prefill but SLO=TPOT) |

**`uses_ttft_slo` property**: Returns `is_prefill` if `slo_is_ttft is None`, otherwise returns `slo_is_ttft`. This decouples the **compute type** (prefill vs decode latency model) from the **SLO type** (TTFT vs TPOT weight). A preempted decode request does context recomputation (prefill compute) but is still measured against its TPOT SLO.

**`baseline_reward()`**: `r_n = w_n · w_TTFT` if `uses_ttft_slo` else `w_n · w_TPOT`.

##### Mathematical Formulation

All time arithmetic inside the solver is in **SECONDS**. At the boundary with `energy_model` (which returns ms) and `ReqView` (which carries `deadline_ms` / `wait_ms`), there is exactly one `/1000` conversion at ingest and one `×1000` on the returned `et_pred`.

Key quantities per request `n` at iteration `i`:

| Symbol | Definition | Units |
|---|---|---|
| `r_n` | `w_n · w_TTFT` (if `uses_ttft_slo`) or `w_n · w_TPOT` (otherwise) — baseline reward | dimensionless |
| `s_n` | `deadline_n − T_{i,n}` — slack (positive = on time, negative = overdue) | seconds |
| `ℓ_{i,n}` | per-iteration token cost: `num_prompt_tokens` for prefill, `1` for decode | tokens |
| `ET_i(B, f)` | predicted batch execution time at frequency `f` | seconds |
| `P(f)` | GPU power draw at frequency `f` | watts |

**Four solver modes** (`SOLUTION_MODE`):

##### H2 (`SOLUTION_MODE=1`): Frequency-independent priority

Frequency-independent single-order admission with joint prefix×frequency enumeration. Uses raw slack `s_n` (not normalized). Complexity: `O(N log N + |F|·|B̂|²)`. Brief overview only — not the active mode.

##### H3 (`SOLUTION_MODE=2`): Frequency-dependent priority

Frequency-dependent priority with per-frequency greedy admission and prefix enumeration. Uses raw slack `s_n`. Complexity: `O(|F|·(N log N + |B̂|²))`. Brief overview only — not the active mode.

##### H4 (`SOLUTION_MODE=3`, default): Frequency-dependent priority with normalization

The key innovation of H4 is **normalized slack and overshoot**, which makes the urgency signal scale-invariant across requests with different SLO targets. This is critical now that the workload uses discrete SLO classes (strict: 600ms TTFT / 80ms TPOT; normal: 1000ms / 100ms; relaxed: 1500ms / 150ms) — without normalization, strict requests with small absolute slack would dominate the priority ordering.

**Step 1 — Normalized Priority (per frequency)**:
```
s̃_n = s_n / deadline_n          (normalized slack: dimensionless)
q_n(f) = [r_n · min(exp(−s̃_n), CAP) − β · P(f) · t_n(f)] / ℓ_n
```

Where:
- `s_n = deadline_s − wait_s` is the raw slack in seconds
- `deadline_n` is the request's SLO target (TTFT or TPOT, depending on `uses_ttft_slo`)
- `t_n(f)` is the per-request latency contribution at frequency `f` (prefill or decode model)
- `CAP = 200000.0` caps deeply-overdue urgency
- `ℓ_n = max(l_q, 1)` converts to value-density per token

Normalization by `deadline_n` ensures that a request with TTFT SLO = 1000ms at 500ms wait has the same urgency as a request with TPOT SLO = 100ms at 50ms wait (both at 50% slack). The frequency-dependent energy term `β · P(f) · t_n(f)` subtracts the marginal energy cost of adding request `n` at frequency `f`.

Implementation details (solver.py L464–638):
- Vectorized computation: `is_pf`, `uses_ttft_slo`, `is_waiting`, `l_q`, `l_kv`, `w_n`, `deadline_s`, `wait_s` are extracted into numpy arrays from the `ReqView` list.
- `r_n_vec = w_n * np.where(uses_ttft_slo, cfg.w_ttft, cfg.w_tpot)` — baseline reward uses the `uses_ttft_slo` property, not `is_prefill`.
- Prefill latency contribution: `wp_contrib = a_p · l_q² + b_p · l_q · l_kv + c_p · l_q` (zero for decode)
- Decode latency contribution: `wd_contrib = a_d · l_kv + b_d` (zero for prefill)
- Per-request time at each frequency: `t_nf = (wp_contrib / f + wd_contrib / f^α) / 1000` (ms→s)
- Priority matrix: `q_all[fi, n] = (RU[n] − β · P(f) · t_nf[fi, n]) / ℓ_n` where `RU = r_n · urgency`
- `orders_all = np.argsort(-q_all, axis=1)` — separate sorted order per frequency

**Step 2 — Greedy Fill with q_n ≤ 0 Cutoff (per frequency)**:

For each frequency candidate `f`:
- Sort requests by `q_n(f)` descending.
- Admit requests while ALL of:
  - `|B| < B_max` (batch size cap)
  - `cum_tokens ≤ L_max` (token budget)
  - `q_n(f) > 0` ← **key cutoff**: requests whose marginal energy cost exceeds their marginal utility are never admitted
  - For waiting requests: `used_waiting < waiting_capacity` (= `max_num_running_reqs − len(running)`)
- **Chunked prefill handling** (`IS_CHUNKED_PREFILL=1`): When a prefill request would exceed `L_max`, it is admitted with a truncated `l_q = L_max − used_tok` instead of being skipped. The latency contribution `wp_contrib` is recomputed with the truncated `l_q`. The truncated `l_q` is stored in a `chunked_override` dict and applied back to the `ReqView` after the best frequency is selected. After truncation, the greedy fill `break`s (no more requests can fit).

**Step 3 — Batch Evaluation (no prefix enumeration)**:

Unlike H2/H3, H4 does **not** enumerate prefix subsets. The full greedy batch for each frequency is evaluated as a single batch:
```
ET_i(B, f) = (Σ_p wp_n/f + Σ_d wd_n/f^α + ovh(B,f) + t_c) / 1000

ovh(B, f) = w_pf/f  (if batch has any prefill) + w_dec/f^α  (if batch has any decode)

overshoot_n = max(ET_i(B, f) − s_n, 0) / deadline_n    (normalized)

J(f) = Σ_{n∈B} r_n · exp(−overshoot_n) − β · P(f) · ET_i(B, f)

f* = argmax_f J(f)
```

The normalized overshoot `/ deadline_n` in the utility function ensures that a 10ms overshoot on a 100ms TPOT SLO penalises the same as a 100ms overshoot on a 1000ms TTFT SLO (both 10%).

**Key differences from H2/H3**: (1) normalized slack in priority, (2) q_n ≤ 0 cutoff, (3) no prefix enumeration, (4) normalized overshoot in utility, (5) waiting_capacity admission constraint, (6) chunked prefill truncation.

**Frequency stride**: All modes subsample frequency candidates by `freq_stride`. A800 has ~82 supported SM clocks (210–1410 MHz, 15 MHz steps); with `freq_stride=3`, the solver evaluates every 3rd clock (~28 candidates). If the max frequency (1410 MHz) is not in the subsampled list, it is appended as a fallback. When `β=0`, only 1410 MHz is evaluated (no energy saving possible).

##### H5 (`SOLUTION_MODE=4`): Incremental marginal-value cutoff

Similar to H4 in using normalized slack and no prefix enumeration, but replaces the batch-level cutoff `q_n(f) > 0` with an **incremental marginal-value test** Δ_n. During greedy fill, the running batch execution time `et_hat_s` is tracked incrementally; each candidate request's overshoot is evaluated against the *current* batch time (not just its own latency), and admission stops when the marginal utility-minus-energy contribution turns negative. Brief overview only — not the active mode.

#### 3.9.2 vLLM Integration — Embedded `_schedule_energy()`

The energy scheduling logic is embedded directly in vLLM's `Scheduler` class via a git patch (`scheduler_energy.patch`). Controlled by `VLLM_ENERGY_SCHEDULER=1` env var — when disabled (default), zero overhead.

**`__init__` additions** (L298–347): Loads `EnergySchedConfig`, latency/power models (`_elat`, `_epow`), frequency controller (`_efreq`), creates solver instance (`_esolver`). Initialises:
- `_eiter`: iteration counter
- `_elog`: iteration log file handle
- `_eprev_exit_t`, `_eprev_record`: for exec_ms timing
- `_elast_exec`: per-request last execution timestamp (ms)
- `_elast_output_ms`: per-request last **real output** timestamp (ms) — distinct from `_elast_exec`, which also covers recomputation steps
- `_ereq_state`: per-request state for online weight update
- `_epreempt_cooldown`: preemption cooldown timestamps (mode 1)
- `_epreempt_multiplier`: preemption priority multipliers (mode 2)

**`schedule()` dispatch**:
```python
def schedule(self) -> SchedulerOutput:
    if self._energy_enabled:
        return self._schedule_energy()
    # ... default path unchanged ...
```

**`_schedule_energy()`** — 10 phases:

| Phase | Description |
|-------|-------------|
| 0 | Timing + release preempt cooldowns + online SLO-weight update (`_energy_update_weights`) |
| 1 | Build `ReqView` list from waiting + running → compute `effective_Lmax` (min of solver Lmax and 95% of free KV tokens) and `waiting_capacity` (max_num_running_reqs − len(running)) → call `solver.solve()` |
| 2 | Init scheduling state (lists, dicts, `kv_cache_manager.new_step_starts()`) |
| 3 | Classify chosen into running vs waiting (if chosen non-empty) |
| 4 | Schedule chosen RUNNING requests (direct `allocate_slots`, token_budget tracking; skip on allocation failure) |
| 5 | Admit chosen WAITING requests: respects `waiting_capacity` and `max_num_running_reqs` as admission cap only (never triggers preemption); calls `get_computed_blocks` + `allocate_slots` |
| 6 | Set GPU frequency AFTER batch confirmed (max freq if nothing scheduled) |
| 7 | Emergency preemption: only triggers on KV deadlock (see §3.9.4) |
| 8 | Construct `SchedulerOutput` (handles `use_v2_model_runner`, calls `_update_after_schedule`) |
| 9 | Logging + increment `_eiter` |

**Key design principles**:
- **Admission-only cap**: `max_num_running_reqs` is respected as an admission gate — waiting requests are not admitted beyond this cap, but running requests are never preempted to make room for new admissions.
- **KV-aware Lmax**: `effective_Lmax = min(solver_Lmax, 0.95 × free_kv_tokens)` prevents the solver from choosing a batch that cannot be allocated.
- **Single `allocate_slots` per request**: No double KV check.
- **GPU frequency set AFTER batch confirmed**: Not before.

**Helper methods**:

- **`_energy_build_views(now_ms)`** (L1376–1460): Converts vLLM requests to `ReqView`. Key logic:
  - **Waiting requests**: In cooldown mode (IS_COOLDOWN=1), skips requests in `_epreempt_cooldown`. In decay mode (IS_COOLDOWN=2), applies `_energy_effective_w_n()` to reduce priority. Distinguishes fresh prefill (never produced output) from **preempted decode recomputation** (has `num_output_tokens > 0`): the latter sets `is_prefill=True` (uses prefill latency model) but `slo_is_ttft=False` (SLO weight = TPOT, deadline = TPOT). Wait time for preempted decode uses `_energy_tpot_wait_ms()`.
  - **Running requests**: Handles three states: (a) partial prefill (chunked, `num_computed < num_prompt_tokens`), (b) resumed decode doing context recomputation (`num_output > 0` but `num_computed < recompute_target`), (c) normal decode. For resumed decode recomputation, `l_q` is set to all remaining tokens (not 1-per-step) and `slo_is_ttft=False`.

- **`_energy_tpot_wait_ms(rid, now_ms)`** (L1462–1468): Returns `now_ms − last_output_ms`, where `last_output_ms` is the wall-clock time of the most recent **real** output token (recorded in `update_from_output()`). Falls back to `_elast_exec` if no output recorded yet. This is distinct from `_elast_exec` because recomputation steps must not reset the TPOT clock.

- **`_energy_effective_w_n(request_id, original_w_n)`** (L1470–1473): In decay mode (IS_COOLDOWN=2), returns `original_w_n × preempt_multiplier`. Otherwise returns `original_w_n` unchanged.

- **`_energy_get_arrival(req, now_ms)`** (L1475–1484): Priority: `send_time` from `vllm_xargs` > `req.arrival_time` > `now_ms`.

- **`_energy_extract_slos(req)`** (L1486–1494): Pulls `(ttft_slo_ms, tpot_slo_ms, w_n)` from request's `extra_args`.

- **`_energy_ensure_req_state(rid, req, now_ms)`** (L1496–1509): Initialises per-request state for online weight update on first sight.

- **`_energy_update_weights(now_ms)`** (L1511–1560): Online adaptive update — see §3.9.3.

- **`_energy_release_preempt_cooldowns(now_ms)`** (L1350–1374): Only active in cooldown mode (IS_COOLDOWN=1). Releases requests from cooldown whose TTFT SLO duration has elapsed since preemption. In decay mode, clears the cooldown dict entirely (a no-op guard).

**`update_from_output()` addition** (L2016–2018): When energy scheduling is enabled, records `_elast_output_ms[req_id] = time.time() * 1000.0` on every real output token. This is the ground truth for TPOT wait calculation — recomputation steps do not advance it.

#### 3.9.3 Online Adaptive Weight Update

Implemented in `_energy_update_weights()`, runs at the start of every `schedule()` call:

- **TTFT update** (triggered when first output token appears, `n_out >= 1` and `ttft_fired == False`):
  ```
  ttft_ratio = (now_ms − arrival_ms) / ttft_slo_ms
  ttft_pos_viol = max(0, ttft_ratio − 1.0)
  w_TTFT ← max(0.01, w_TTFT + η_TTFT · w_n · (ttft_pos_viol − 0.05))
  ```
  The `−0.05` term acts as a target margin: the weight decreases slightly even when the SLO is met (but by less than 5% margin), providing natural relaxation when the system is well within SLO bounds.

- **TPOT update** (triggered when request disappears from running+waiting, i.e., completes):
  ```
  avg_tpot_obs = (now_ms − first_tok_ms) / (n_decode_tokens)
  tpot_ratio = avg_tpot_obs / tpot_slo_ms
  tpot_pos_viol = max(0, tpot_ratio − 1.0)
  w_TPOT ← max(0.01, w_TPOT + η_TPOT · w_n · (tpot_pos_viol − 0.05))
  ```

Note: `η_TTFT` and `η_TPOT` default to `0.0` in `EnergySchedConfig`, meaning the weight update is disabled unless explicitly set via environment variables. The initial `W_TTFT=1000` and `W_TPOT=100` values are therefore static throughout the experiment under current defaults.

#### 3.9.4 Preemption Handling — Decay Mode (`IS_COOLDOWN=2`, default)

Emergency preemption only triggers on **KV deadlock**: when KV allocation failed AND nothing was scheduled AND the running queue is non-empty. The scheduler preempts the lowest-progress request (`min(running, key=num_computed_tokens)`).

In decay mode (`IS_COOLDOWN=2`), preempted requests are handled as follows:

1. **Preemption event** (Phase 7, L1193–1201): The request's scheduler-local priority multiplier is divided by `DECAY_PARAMETER` (default 10000):
   ```python
   prev = self._epreempt_multiplier.get(vid, 1.0)
   self._epreempt_multiplier[vid] = max(prev / decay, min_mult)
   ```
   With `DECAY_PARAMETER=10000` and `min_mult=0.000005`, a single preemption reduces the effective priority to 0.0001× of the original. This is an extreme demotion — the request will rank near the bottom of the priority ordering.

2. **Effect on solver**: When `_energy_build_views()` constructs `ReqView` objects, it calls `_energy_effective_w_n(rid, original_w_n)` which returns `original_w_n × multiplier`. Since `q_n(f) = [r_n · urgency − β · P(f) · t_n(f)] / ℓ_n` and `r_n = effective_w_n · w_TTFT_or_TPOT`, the drastically reduced `w_n` makes the request's `q_n(f)` near-zero or negative, causing the H4 cutoff to exclude it from the batch.

3. **Recovery**: When the preempted request is eventually re-admitted (selected by the solver despite low priority), `_epreempt_multiplier` is cleared on admission (L1169). The request then runs at full priority again.

4. **Cleanup**: When a request completes (disappears from running+waiting), the multiplier entry is removed (L1545).

**Contrast with cooldown mode** (`IS_COOLDOWN=1`): In cooldown mode, preempted requests are completely hidden from the solver (excluded from `_energy_build_views()`) until their TTFT SLO duration has elapsed since preemption. This is a hard exclusion — the request is invisible. Decay mode is softer: the request remains visible but strongly deprioritized, allowing the solver to include it if nothing better is available.

#### 3.9.5 Chunked Prefill Integration (`IS_CHUNKED_PREFILL=1`, default)

When chunked prefill is enabled (`--enable-chunked-prefill` on the vLLM server, `IS_CHUNKED_PREFILL=1` in main.sh):

**Solver side** (all modes H2–H5):
- During greedy fill, if a prefill request's `l_q` would exceed the remaining token budget (`used_tok + tok_n > Lmax`), it is admitted with a **truncated** token count: `tok_n = Lmax − used_tok`.
- The latency contribution `wp_contrib` is recomputed using the truncated `l_q` and the actual `l_kv`: `wp_n = a_p · l_q² + b_p · l_q · l_kv + c_p · l_q`.
- The truncation is recorded in `chunked_override[position] = (new_wp, truncated_l_q)` and applied back to the `ReqView.l_q` after the best frequency is selected. The scheduler uses this truncated `l_q` as `num_new_tokens` when calling `allocate_slots()`.
- After admitting a chunked request, the greedy fill `break`s (token budget is exhausted).

**Scheduler side** (`_energy_build_views`, L1376–1460):
- A running request with `num_computed < num_prompt_tokens` (partial prefill) is a **chunked prefill in progress**. It appears as `is_prefill=True` with `l_q = remaining = num_prompt − num_computed` and `l_kv = num_computed`.
- This correctly uses the cross-term `b_p · l_q · l_kv` in the latency model, which captures the attention computation between the new prompt chunk and the already-computed KV context.
- A waiting request that was preempted after partial prefill but before any output (`num_output == 0`) also appears with the remaining prompt tokens.

**Phase 5 — Waiting request admission** (L1109–1171): Token budget is enforced per vLLM's chunked-prefill policy. With `enable_chunked_prefill=True`, `num_new_tokens = min(v.l_q, req.num_tokens − num_computed, token_budget)`. Without chunked prefill, a request that exceeds `token_budget` causes a `break`.

---

### 3.10 `vllm_patches/apply_patch.sh` (60 lines) — Patch installer

1. Clears Python bytecode cache in `vllm/energy_sched/`
2. Copies `solver.py`, `energy_model.py`, `frequency_controller.py`, `__init__.py` into `vllm/energy_sched/`
3. Removes old monkey-patch hook from `vllm/__init__.py` if present
4. Applies `scheduler_energy.patch` via `git apply`
5. If patch is already applied, skips. If patch is stale (old version detected), extracts clean scheduler from git HEAD, applies patch to that, and overwrites (with backup).

### 3.11 `vllm_patches/unapply_patch.sh` (40 lines) — Patch rollback

1. Reverses `scheduler_energy.patch` via `git apply -R` (falls back to `git checkout` on failure)
2. Removes `vllm/energy_sched/` directory
3. Removes old monkey-patch hook if still present

### 3.12 `vllm_patches/scheduler_energy.patch` — Git patch for scheduler.py

Generated via `git diff` from the vLLM repo. Adds the `_energy_enabled` init block and energy scheduling methods to `Scheduler`. Also adds `prev_step_scheduled_req_ids.discard()` to `_preempt_request` to prevent stale request IDs from persisting after preemption, and records `_elast_output_ms` in `update_from_output()`.

---

## 4. vLLM Edits

**File**: `vllm/v1/core/sched/scheduler.py` (modified via `scheduler_energy.patch`)

**Changes**:
1. `__init__` (L298–347): Adds `self._energy_enabled` flag and energy scheduler init (solver, frequency controller, latency/power models, preempt cooldown/multiplier dicts, last-output-ms tracking) — gated by `VLLM_ENERGY_SCHEDULER=1` env var.
2. `schedule()` (L400–401): Adds `if self._energy_enabled: return self._schedule_energy()` dispatch at the top — default path is completely unchanged.
3. `_preempt_request()` (L1583): Adds `self.prev_step_scheduled_req_ids.discard(request.request_id)` to prevent stale IDs after preemption.
4. `update_from_output()` (L2016–2018): Records `_elast_output_ms[req_id]` on every real output token for accurate TPOT wait tracking.
5. New energy methods: `_schedule_energy`, `_energy_release_preempt_cooldowns`, `_energy_build_views`, `_energy_tpot_wait_ms`, `_energy_effective_w_n`, `_energy_get_arrival`, `_energy_extract_slos`, `_energy_ensure_req_state`, `_energy_update_weights`.

No changes to `vllm/__init__.py` (old monkey-patch hook removed).

## 5. Dataset Provenance

- **Repo**: `RyokoAI/ShareGPT52K` on Hugging Face
- **Auto-download**: `prepare_dataset.py` automatically downloads the dataset via `huggingface_hub` if the directory is missing, or re-downloads if Git LFS pointers are detected.
- **Trace**: 1000 requests (default), first human message per conversation, prompt length 512–6000 chars
- **Output tokens**: tokenized from reference assistant outputs (actual token counts, not random sampling)
- **SLO parameters**: discrete classes — strict (30%: TTFT=600ms, TPOT=80ms), normal (50%: TTFT=1000ms, TPOT=100ms), relaxed (20%: TTFT=1500ms, TPOT=150ms)
- **Arrival rate**: 3 req/s (uniform)

## 6. Results (latest run)

Parameters: `BETA=1.0, W_TTFT=1000 (initial), W_TPOT=100 (initial), SOLUTION_MODE=3 (H4), DEFAULT_MAX_NUM_SEQS=256, CUSTOM_MAX_NUM_SEQS=400, NUM_REQUESTS=400, RATE_QPS=3, FREQ_STRIDE=3, MAX_BATCH_SIZE=256, IS_COOLDOWN=2, DECAY_PARAMETER=10000, IS_CHUNKED_PREFILL=1, SOLVER_LMAX=8192, VLLM_MAX_BATCHED_TOKENS=8192, GPU_MEM_UTIL=0.95`.

| Metric | Default | Custom |
|--------|---------|--------|
| mean_ttft_ms | 120.65 | 122.57 |
| mean_tpot_ms | 44.79 | 46.02 |
| mean_ttft_violation_ms | 0.0 | 0.0 |
| mean_tpot_violation_ms | 0.08 | 0.15 |
| mean_normalized_ttft_violation | 0.0 | 0.0 |
| mean_normalized_tpot_violation | 0.001503 | 0.003017 |
| ttft_slo_attainment | 1.0 | 1.0 |
| tpot_slo_attainment | 0.945 | 0.93 |
| mean_power_w | 339.86 | 349.71 |
| total_energy_j | 57132.36 | 59000.18 |
| mean_solve_exec_ratio | 0.0 | 0.008139 |

**Notes on results**:
- All 400 requests completed in both modes.
- **TTFT**: Both schedulers achieve 100% TTFT SLO attainment. Custom scheduler has near-identical mean TTFT (122.6ms vs 120.7ms).
- **TPOT**: Mean TPOT increases slightly (46.0ms vs 44.8ms). TPOT SLO attainment drops marginally (93% vs 94.5%). Normalized TPOT violation is very low in both modes (0.003 vs 0.0015).
- **Energy**: With `β=1.0`, the solver does not aggressively lower GPU frequency, resulting in similar power (349.7W vs 339.9W) and energy (59.0kJ vs 57.1kJ). Higher β values (e.g., β=3.0) produce more aggressive energy savings at the cost of increased SLO violations.
- **Solver overhead**: `mean_solve_exec_ratio = 0.008` means the solver takes ~0.8% of batch execution time — negligible overhead for the H4 heuristic.

## 7. How to Reproduce

### Prerequisites

- NVIDIA GPU (tested on A800-SXM4-80GB)
- Conda environment `myvllm` with vLLM installed from source at `/home/ubuntu/lqs/vllm`
- Model weights at `/home/ubuntu/lqs/LLM_model` (Qwen3-14B)
- `sudo` access (required for GPU frequency locking via `pynvml` / `nvidia-smi`)

### Quick Start

```bash
cd /home/ubuntu/lqs/energy_efficient_LLM_scheduling

# Run the full experiment (patch + trace + baseline + custom + comparison)
sudo bash main.sh

# View results
cat results/demo/comparison.csv
```

`main.sh` handles everything end-to-end:
1. Applies the vLLM patch (`apply_patch.sh` → copies `energy_sched/` package + applies `scheduler_energy.patch`)
2. Generates `trace.jsonl` from ShareGPT52K (auto-downloads if needed)
3. Runs baseline experiment (`VLLM_ENERGY_SCHEDULER=0`)
4. Runs custom experiment (`VLLM_ENERGY_SCHEDULER=1`)
5. Compares results → `results/${TAG}/comparison.csv`

### Manual Step-by-Step

```bash
cd /home/ubuntu/lqs/energy_efficient_LLM_scheduling
conda activate myvllm

# 1. Generate the workload trace (only needed once)
python scripts/prepare_dataset.py

# 2. Apply the vLLM patch
bash vllm_patches/apply_patch.sh /home/ubuntu/lqs/vllm

# 3. Run the experiment
sudo bash main.sh

# 4. (Optional) Rollback the patch
bash vllm_patches/unapply_patch.sh /home/ubuntu/lqs/vllm
```

### Configuration

Edit the **USER KNOBS** block at the top of `main.sh` to change:
- `TAG`: output directory name under `results/`
- `MODE`: `"default"`, `"custom"`, or `"both"`
- `BETA`, `W_TTFT`, `W_TPOT`: energy-utility trade-off parameters
- `SOLUTION_MODE`: `1` (H2), `2` (H3), `3` (H4, default), `4` (H5)
- `IS_COOLDOWN`: `1` (cooldown) or `2` (decay, default)
- `IS_CHUNKED_PREFILL`: `0` (disabled) or `1` (enabled, default)
- `VLLM_MAX_BATCHED_TOKENS`, `SOLVER_LMAX`: token budget controls
- See `main.sh` for the full list of knobs

To change trace generation parameters (SLO classes, prompt filters, arrival rate), edit `scripts/prepare_dataset.py` and delete `trace.jsonl` to regenerate.

### Project Structure

```
energy_efficient_LLM_scheduling/
├── main.sh                          # Master experiment runner
├── trace.jsonl                      # Generated workload trace
├── experiment.md                    # This file
├── scripts/
│   ├── prepare_dataset.py           # Trace generator (ShareGPT → trace.jsonl, discrete SLO classes)
│   ├── workload_sender.py           # Async HTTP workload replay
│   ├── power_logger.py              # GPU power sampling (pynvml → CSV)
│   ├── metrics_collector.py         # Results aggregation → summary.json
│   └── compare_results.py           # Side-by-side comparison → CSV
├── vllm_patches/
│   ├── solver.py                    # Pure algorithm (EnergySchedConfig, Alt1HeuristicSolver: H2+H3+H4+H5)
│   ├── energy_model.py              # Latency + power models (LatencyParams, PowerParams)
│   ├── frequency_controller.py      # GPU SM clock control (pynvml wrapper)
│   ├── __init__.py                  # Package re-exports
│   ├── scheduler_energy.patch       # Git patch for vLLM scheduler.py
│   ├── apply_patch.sh               # Patch installer (with stale-patch refresh)
│   └── unapply_patch.sh             # Patch rollback
└── results/
    └── ${TAG}/                      # Output directory (per TAG)
        ├── server_{default,custom}.log
        ├── power_{default,custom}.csv
        ├── results_{default,custom}.jsonl
        ├── summary_{default,custom}.json
        ├── iter_custom.log
        └── comparison.csv
```
