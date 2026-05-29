# Experiment Record — Energy-Efficient LLM Scheduling on vLLM

## 1. Problem Statement

We run a single vLLM server serving Qwen3-14B on an A800-SXM4-80GB GPU and compare two schedulers on the same workload:

- **Baseline**: vLLM's default FCFS scheduler, GPU clocks not locked.
- **Ours (custom)**: an energy-aware scheduler based on the **Heuristic 4 (H4)** formulation — a two-step algorithm (frequency-dependent priority scoring with normalized slack → greedy fill with q_n≤0 cutoff) with online adaptive weight updates. The scheduler selects both the GPU SM frequency and batch composition per iteration, locking the SM clock via `pynvml`.

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

### 3.1 `main.sh` (314 lines) — Master experiment orchestrator

**Purpose**: Controls the full experiment lifecycle — applies the vLLM patch, generates the workload trace, launches the vLLM server, replays the workload, logs power, collects metrics, and compares results.

**L1–123: USER KNOBS block** — All tunable parameters are declared as Bash variables at the top of the file:

| Variable | Default | Meaning |
|---|---|---|
| `TAG` | `"demo"` | Output directory name under `results/` |
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
| `NUM_REQUESTS` | `400` | Number of requests in the workload |
| `RATE_QPS` | `2.0` | Arrival rate (requests/second) |
| `MIN_OUT_TOK` / `MAX_OUT_TOK` | `1024` / `1024` | Output token range per request |
| `TRACE_SEED` | `42` | Random seed for trace generation |
| `BETA` | `3.0` | Energy-utility trade-off (larger = more energy-saving) |
| `W_TTFT` | `2000.0` | Initial weight for TTFT in priority calculation (mutable — drifts online) |
| `W_TPOT` | `1.0` | Initial weight for TPOT in priority calculation (mutable — drifts online) |
| `ETA_MS` | `200` | Per-iteration time budget η (ms); accepted for compat but currently unused by solver |
| `VLLM_MAX_BATCHED_TOKENS` | `8192` | Passed to vLLM `--max-num-batched-tokens`; must be ≥ MAX_MODEL_LEN |
| `SOLVER_LMAX` | `8192` | Solver-side max tokens per batch in greedy fill (can differ from vLLM batched tokens) |
| `FREQ_STRIDE` | `3` | Stride for frequency candidate subsampling |
| `MAX_BATCH_SIZE` | `256` | Max requests per iteration (batch cap) |
| `SOLUTION_MODE` | `3` | Solver heuristic (1=H2 freq-indep priority, 2=H3 freq-dep priority, 3=H4 freq-dep with normalization) |
| `IS_CHUNKED_PREFILL` | `1` | 0=non-chunked prefill, 1=chunked prefill |
| `POWER_INTERVAL_S` | `0.1` | GPU power sampling interval (seconds) |

**L125**: Captures the script directory so all paths are absolute regardless of CWD.

**L128–138**: Conda activation. Tries four possible `conda.sh` locations (miniconda3, anaconda3, /opt/conda), sources the first one found, then activates the `myvllm` environment.

**L143–144**: Calls `apply_patch.sh` to copy the energy scheduler Python files into the vLLM source tree and apply the scheduler patch.

**L147–152**: Conditional trace generation. If `trace.jsonl` already exists, it is reused. Delete the file to force regeneration.

**L156–165**: `reset_gpu_clocks()` helper — uses `FrequencyController` to reset GPU clocks, falls back to `nvidia-smi -rgc / -rmc`.

**L168–296: `run_experiment()` function** — The core experiment runner:

- **L169–174**: Maps `"default"`/`"custom"` label to the output file suffix.
- **L180–198**: Builds the server environment variable array. For baseline, `VLLM_ENERGY_SCHEDULER=0`. For custom mode, sets `VLLM_ENERGY_SCHEDULER=1` plus all hyperparameters (`VLLM_ENERGY_BETA`, `VLLM_ENERGY_W_TTFT`, `VLLM_ENERGY_W_TPOT`, `VLLM_ENERGY_LMAX`, `VLLM_ENERGY_MAX_BATCH_SIZE`, `VLLM_ENERGY_FREQ_STRIDE`, `VLLM_ENERGY_SOLUTION_MODE`, `VLLM_ENERGY_ETA_MS`, `VLLM_ENERGY_GPU_INDEX`, `VLLM_ENERGY_ITER_LOG`, `VLLM_ENERGY_CHUNKED_PREFILL`).
- **L206–233**: Launches the vLLM server as a background process. Selects `DEFAULT_MAX_NUM_SEQS` or `CUSTOM_MAX_NUM_SEQS` depending on mode. Key flags:
  - `--enforce-eager`: Disables CUDA graphs (needed because frequency changes invalidate graph caches)
  - `--no-async-scheduling`: Disables async scheduling so the scheduler sees all running/waiting requests at each iteration
  - `--enable-chunked-prefill` or `--no-enable-chunked-prefill`: Controlled by `IS_CHUNKED_PREFILL` knob
  - `--max-num-batched-tokens`: Set from `VLLM_MAX_BATCHED_TOKENS` if > 0
  - `--no-enable-prefix-caching`: Disables prefix caching (simplifies KV cache accounting)
  - `--enable-logging-iteration-details`: Enables detailed per-iteration logging in server log
  - stdout/stderr redirected to `server_${label}.log`
- **L236–249**: Health check loop. Polls `http://localhost:PORT/health` every 2 seconds for up to 240 seconds. If the server process dies before becoming ready, prints the log and exits.
- **L252–257**: Starts `power_logger.py` as a background process, sampling GPU power every 0.1s.
- **L260–265**: Runs `workload_sender.py` synchronously — it blocks until all requests are done.
- **L268–270**: Stops the power logger via `kill`.
- **L273–275**: Stops the vLLM server via `kill`.
- **L278–280**: Resets GPU clocks to default after custom mode. Uses the `FrequencyController` Python class first, falls back to `nvidia-smi -rgc / -rmc`.
- **L283–294**: Runs `metrics_collector.py` to aggregate results into `summary_${label}.json`.

**L299–305**: Sequential experiment execution. If `MODE` is `"default"` or `"both"`, runs baseline. If `"custom"` or `"both"`, runs custom.

**L308–312**: Runs `compare_results.py` to produce a side-by-side comparison table and CSV.

---

### 3.2 `scripts/prepare_dataset.py` (149 lines) — Synthetic trace generation

**Purpose**: Ensures the ShareGPT52K dataset is available (auto-downloading if needed, including re-downloading Git LFS pointers), filters and samples prompts, and writes `trace.jsonl` — one JSON record per line representing a single request with its arrival time, prompt, and SLO parameters.

**L17–61: USER KNOBS block**:

| Constant | Default | Meaning |
|---|---|---|
| `OUTPUT` | `"trace.jsonl"` | Output file path |
| `NUM_REQUESTS` | `400` | Number of requests to sample |
| `RATE_QPS` | `4` | Arrival rate — request i arrives at `i / RATE_QPS` seconds |
| `TTFT_MEAN_MS` | `4000.0` | Mean TTFT SLO target (ms) |
| `TTFT_STD_MS` | `800.0` | Std dev of TTFT SLO |
| `TPOT_MEAN_MS` | `100.0` | Mean TPOT SLO target (ms) |
| `TPOT_STD_MS` | `40.0` | Std dev of TPOT SLO |
| `MIN_OUTPUT_TOKENS` / `MAX_OUTPUT_TOKENS` | `64` / `1024` | Output token range |
| `MIN_PROMPT_CHARS` / `MAX_PROMPT_CHARS` | `512` / `8000` | Prompt length filter (characters) |
| `SEED` | `42` | Random seed for reproducibility |
| `DATASET_DIR` | `"data/sharegpt52k"` | Local path to ShareGPT dataset |
| `REPO_ID` | `"RyokoAI/ShareGPT52K"` | Hugging Face repository ID for auto-download |

**L64–83: `_ensure_dataset()`** — Automatic dataset verification:
- If `DATASET_DIR` doesn't exist, downloads via `huggingface_hub.snapshot_download()`.
- Checks each `.json` file for Git LFS pointer signatures (file size < 200 bytes, content starts with `version `). If detected, removes and re-downloads.

**L86–90: `truncated_normal()`**: Samples from a Gaussian distribution and rejects values ≤ `low`. Used to generate TTFT/TPOT SLO targets that are always positive.

**L93–112: Dataset loading**:
- Iterates all `.json` files in `DATASET_DIR`.
- For each conversation, extracts the first human/user message.
- Filters by prompt length (512–8000 characters).
- Handles both ShareGPT format keys (`"from"`/`"value"`) and OpenAI format keys (`"role"`/`"content"`).

**L127–142: Trace writing**:
- Shuffles all candidate prompts, takes the first `NUM_REQUESTS`.
- For each request, writes a JSON record with:
  - `id`: unique identifier like `"req_000001"`
  - `arrival_s`: `i / RATE_QPS` — uniform arrival times (at 4 QPS, requests arrive every 0.25s)
  - `prompt`: the actual text content
  - `max_tokens`: uniformly sampled from `[64, 1024]`
  - `ttft_ms`: sampled from truncated normal (μ=4000, σ=800)
  - `tpot_ms`: sampled from truncated normal (μ=100, σ=40)
  - `w_n`: priority weight, default 1.0

---

### 3.3 `scripts/workload_sender.py` (173 lines) — Async workload replay

**Purpose**: Reads `trace.jsonl` and asynchronously sends each request to the vLLM `/v1/completions` endpoint with `stream=true`, measuring per-request TTFT and TPOT.

**L131–133: Custom TCP connector**: Uses `aiohttp.TCPConnector(limit=1000, limit_per_host=1000)` to raise the connection pool ceiling. The default aiohttp limit (100 total / 0 per-host) can become a bottleneck when hundreds of requests are in-flight concurrently at high QPS, causing artificial queuing at the HTTP layer that inflates measured TTFT. The explicit 1000-connection limit eliminates this bottleneck.

**L19–34: `ResultRecord` dataclass**: Holds per-request metadata and results:
- `id`, `prompt`, `max_tokens`, `ttft_slo_ms`, `tpot_slo_ms`, `w_n`, `arrival_s`
- `send_time`: wall-clock epoch (seconds) recorded just before the HTTP POST (new — see §3.3a below)
- `complete_time`: wall-clock epoch when the request finishes
- `status`, `ttft_ms`, `tpot_ms`, `num_output_tokens`, `error`

**L37–112: `send_one()`**: Sends a single request and measures timing:
- **L55–67**: Builds the HTTP POST payload. Passes TTFT/TPOT/w_n/**send_time** to the server via the `vllm_xargs` field — this is how the energy scheduler receives per-request SLO information and the client-side send timestamp.
- **L53**: Records `rr.send_time = wall_time()` before the HTTP POST.
- **L73–101**: Parses the SSE stream, counting tokens and recording inter-chunk gaps for TPOT computation.
- **L102–103**: TTFT is measured as `(first_chunk_time - rr.send_time) * 1000.0` — relative to **send_time**, not arrival_time.

#### 3.3a Key Design Change: `arrival_time` → `send_time`

In the previous version, TTFT was measured against vLLM's internal `req.arrival_time`, which is set inside `input_processor.process_inputs`. Under high load, the engine event loop may be busy, causing `arrival_time` to be set *after* significant queuing delay has already elapsed. This means `arrival_time` underestimates the actual waiting time experienced by the client.

**The fix**: The workload sender now records a `send_time` (wall-clock epoch in seconds) *before* the HTTP POST and passes it to the server via `vllm_xargs.send_time`. The energy scheduler's `_energy_get_arrival()` method (see §3.9) uses `send_time` as the authoritative arrival time, falling back to `req.arrival_time` only if `send_time` is absent. TTFT measurement in the sender also uses `send_time` as the baseline. This captures the full end-to-end latency including HTTP and engine-queue delays.

**L115–169: `main()`**: Orchestrates the workload replay:
- Loads all requests from `trace.jsonl`.
- Dispatches requests respecting arrival times using `asyncio.sleep`.
- Writes `results.jsonl` with all per-request metrics including `send_time` and `complete_time`.

---

### 3.4 `scripts/power_logger.py` (58 lines) — GPU power sampling

**Purpose**: Continuously samples GPU power draw, SM clock frequency, and GPU utilization via `pynvml`, writing a CSV row every 0.1 seconds.

**L20–22: Signal handler**: Sets a global `_stop` flag on SIGTERM/SIGINT for clean shutdown.

**L28: CSV columns**: `timestamp_s, power_w, sm_clock_mhz, utilization_pct`

**L30–52: Main sampling loop**:
- Calls `nvmlDeviceGetPowerInfo()` to get power in milliwatts, converts to watts.
- Calls `nvmlDeviceGetClockInfo()` for SM clock frequency.
- Calls `nvmlDeviceGetUtilizationRates()` for GPU utilization percentage.
- Writes a CSV row with `flush=True` after each sample.
- Sleeps for the configured interval (0.1s by default).

---

### 3.5 `scripts/metrics_collector.py` (192 lines) — Metrics aggregation

**Purpose**: Reads `results.jsonl` and `power.csv` from a completed experiment, computes summary statistics, and writes `summary.json`.

**L10–16: `trapz()`**: Trapezoidal integration function. Given arrays of time (seconds) and power (watts), computes total energy in joules.

**L19–38: `interpolate_power()`**: Linear interpolation of the power trace at an arbitrary timestamp. Used to compute exact power values at the boundaries of the active window.

**L41–61: `windowed_energy()`**: Computes energy and mean power over a specific time window `[start_ts, end_ts]`:
- Interpolates power at window boundaries via `interpolate_power()`.
- Extracts power samples that fall within the window.
- Integrates energy over this windowed sub-trace via `trapz()`.

This is a significant improvement over the previous approach that integrated over the entire power log duration. The window is defined as `[first_send_time, last_complete_time]` — the active period when requests were actually being processed. This eliminates idle-period power from inflating the energy measurement.

**L64–76: `solve_exec_ratio()`**: Reads the iteration log and returns mean of `solve_ms / exec_ms`.

**L79–191: `main()`**:
- Filters completed requests (HTTP 200, no error).
- Computes mean TTFT, mean TPOT, SLO violations (absolute and **normalized**), and attainment rates.
- **Normalized violations**: `max(0, obs - slo) / slo` — this measures the fractional overshoot relative to each request's SLO target, giving a scale-invariant metric. Requests with tight SLOs are not penalised disproportionately.
- **L143–150**: Extracts `first_send = min(send_time)` and `last_complete = max(complete_time)` across all requests — these define the active window for energy computation.
- **L152–158**: Calls `windowed_energy()` to compute energy only over the active period.
- Writes `summary.json` with all metrics.

---

### 3.6 `scripts/compare_results.py` (65 lines) — Result comparison

**Purpose**: Reads two `summary.json` files (default and custom), prints a side-by-side table, and writes `comparison.csv`. Includes normalized violation metrics (`mean_normalized_ttft_violation`, `mean_normalized_tpot_violation`).

---

### 3.7 `vllm_patches/energy_model.py` (114 lines) — Latency and power models

**Purpose**: Provides the mathematical models for per-request latency and GPU power as functions of frequency. These are used by the scheduler to predict execution time and energy consumption.

**`LatencyParams` dataclass**: 9 Route B+ coefficients fitted to A800-SXM4-80GB profiling data:
- `a_p, b_p, c_p`: prefill latency quadratic model coefficients
- `w_pf, w_dec`: batch overhead weights for prefill and decode
- `a_d, b_d`: decode latency coefficients
- `alpha`: frequency scaling exponent for decode (≈ 0.974)
- `t_c`: constant communication overhead (≈ 4.65 ms)

**`PowerParams` dataclass**: Cubic power model coefficients `(k3, k2, k1, k0)`. The `power_watts(f)` method evaluates `P(f) = k3·f³ + k2·f² + k1·f + k0`.

**`per_request_time_ms()`** — Per-request latency contribution:
- **Prefill**: `t_q = (a_p · l_q² + b_p · l_q · l_kv + c_p · l_q) / f`
- **Decode**: `t_q = (a_d · l_kv + b_d) / f^α`

**`batch_overhead_ms()`** — Mode-dependent batch overhead:
- `T_ovh = I_p · w_pf/f + I_d · w_dec/f^α`

**`batch_time_ms()`** — Total iteration time:
- `ET_i(B, f) = Σ_{n∈B} t_q(n, f) + T_ovh(B, f) + t_c`

---

### 3.8 `vllm_patches/frequency_controller.py` (153 lines) — GPU frequency control

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

#### 3.9.1 `vllm_patches/solver.py` (614 lines) — Algorithm Layer

Contains `EnergySchedConfig`, `ReqView`, `Alt1HeuristicSolver` (H2 + H3 + H4), `baseline_reward()`, `_open_iter_log()`.

##### `ReqView` dataclass

| Field | Type | Meaning |
|---|---|---|
| `handle` | Any | vLLM request object |
| `is_prefill` | bool | True if prefill phase |
| `l_q` | int | Per-iter token cost (prompt tokens for prefill, 1 for decode) |
| `l_kv` | int | KV cache length (computed tokens) |
| `wait_ms` | float | Time since arrival/last execution (ms) |
| `deadline_ms` | float | SLO deadline (TTFT for prefill, TPOT for decode, in ms) |
| `w_n` | float | Per-request priority weight |
| `is_waiting` | bool | True if request is in the waiting queue (used by H4 for admission cap) |
| `kv_blocks_needed` | int | Full KV size in blocks |
| `kv_blocks_incremental` | int | New blocks needed this iteration |

##### Mathematical Formulation

All time arithmetic inside the solver is in **SECONDS**. At the boundary with `energy_model` (which returns ms) and `ReqView` (which carries `deadline_ms` / `wait_ms`), there is exactly one `/1000` conversion at ingest and one `×1000` on the returned `et_pred`.

Key quantities per request `n` at iteration `i`:

| Symbol | Definition | Units |
|---|---|---|
| `r_n` | `w_n · w_TTFT` (prefill) or `w_n · w_TPOT` (decode) — baseline reward | dimensionless |
| `s_n` | `deadline_n − T_{i,n}` — slack (positive = on time, negative = overdue) | seconds |
| `ℓ_{i,n}` | per-iteration token cost: `num_prompt_tokens` for prefill, `1` for decode | tokens |
| `ET_i(B, f)` | predicted batch execution time at frequency `f` | seconds |
| `P(f)` | GPU power draw at frequency `f` | watts |

**Three solver modes** (`SOLUTION_MODE`):

##### H2 (`SOLUTION_MODE=1`): Frequency-independent priority

**Step 1 — One-Shot Priority**:
```
q_n = r_n · min(exp(−s_n), CAP) / ℓ_{i,n}
```
`CAP = 200000.0` caps deeply-overdue urgency. `/ℓ_n` converts to value-density per token.

**Step 2 — Density-Greedy Fill**: Sort by `q_n` descending, admit while `cum_tokens ≤ L_max` and `|B| < B_max`. Produces greedy batch `B̂`.

**Step 3 — Joint (Prefix, Frequency) Enumeration**:
```
(j*, f*) = argmax_{j,f}  Σ_{n∈B_j} r_n · exp(−[ET_i(B_j, f) − s_n]₊) − β · P(f) · ET_i(B_j, f)
```
Vectorised via `np.cumsum` over nested prefixes `B_1 ⊂ B_2 ⊂ ... ⊂ B_{|B̂|}`. Single admission order for all frequencies. Complexity: `O(N log N + |F|·|B̂|²)`.

##### H3 (`SOLUTION_MODE=2`): Frequency-dependent priority

**Step 1 — Frequency-Dependent Priority**:
```
q_n(f) = [r_n · min(exp(−s_n), CAP) − β · P(f) · t_n(f)] / ℓ_n
```
Separate greedy batch per frequency.

**Steps 2–3**: Same prefix enumeration as H2 but per frequency. Complexity: `O(|F|·(N log N + |B̂|²))`.

##### H4 (`SOLUTION_MODE=3`, default): Frequency-dependent priority with normalization

The key innovation of H4 is **normalized slack and overshoot**, which makes the urgency signal scale-invariant across requests with different SLO targets.

**Step 1 — Normalized Priority**:
```
s̃_n = s_n / deadline_n          (normalized slack: dimensionless)
q_n(f) = [r_n · min(exp(−s̃_n), CAP) − β · P(f) · t_n(f)] / ℓ_n
```
Normalization by `deadline_n` ensures that a request with TTFT SLO = 4s at 2s wait has the same urgency as a request with TPOT SLO = 100ms at 50ms wait (both at 50% slack).

**Step 2 — Greedy Fill with q_n ≤ 0 Cutoff**:
- Sort by `q_n(f)` descending per frequency.
- Admit while `cum_tokens ≤ L_max`, `|B| < B_max`, and **`q_n(f) > 0`**.
- The q_n ≤ 0 cutoff means requests whose marginal energy cost exceeds their marginal utility are never admitted — the batch may be smaller than Bmax.
- Waiting requests are additionally capped by `waiting_capacity` (= `max_num_running_reqs − len(running)`).

**Step 3 — No Prefix Enumeration**:
Unlike H2/H3, H4 does **not** enumerate prefix subsets. The full greedy batch is evaluated as a single batch:
```
ET_i(B, f) = (Σ_p wp_n/f + Σ_d wd_n/f^α + ovh(B,f) + t_c) / 1000
overshoot_n = max(ET_i(B, f) − s_n, 0) / deadline_n    (normalized)
J(f) = Σ_{n∈B} r_n · exp(−overshoot_n) − β · P(f) · ET_i(B, f)
f* = argmax_f J(f)
```
The normalized overshoot `/ deadline_n` in the utility function ensures that a 10ms overshoot on a 100ms TPOT SLO penalises the same as a 400ms overshoot on a 4000ms TTFT SLO (both 10%).

**Key differences from H2/H3**: (1) normalized slack in priority, (2) q_n ≤ 0 cutoff, (3) no prefix enumeration, (4) normalized overshoot in utility, (5) waiting_capacity admission constraint.

**Frequency stride**: All modes subsample frequency candidates by `freq_stride`. If the max frequency (1410 MHz) is not in the subsampled list, it is appended as a fallback.

#### 3.9.2 vLLM Integration — Embedded `_schedule_energy()`

The energy scheduling logic is embedded directly in vLLM's `Scheduler` class via a git patch (`scheduler_energy.patch`). Controlled by `VLLM_ENERGY_SCHEDULER=1` env var — when disabled (default), zero overhead.

**`__init__` additions**: Loads `EnergySchedConfig`, latency/power models, frequency controller, creates solver instance. Initialises `_epreempt_cooldown` dict for preemption cooldown tracking.

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
| 7 | Emergency preemption: only triggers on KV deadlock (allocation failed AND nothing scheduled AND running queue non-empty); preempts lowest-progress request and adds to `_epreempt_cooldown` — does NOT retry scheduling in the same step |
| 8 | Construct `SchedulerOutput` (handles `use_v2_model_runner`, calls `_update_after_schedule`) |
| 9 | Logging + increment `_eiter` |

**Key design principles**:
- **Admission-only cap**: `max_num_running_reqs` is respected as an admission gate — waiting requests are not admitted beyond this cap, but running requests are never preempted to make room for new admissions.
- **KV-aware Lmax**: `effective_Lmax = min(solver_Lmax, 0.95 × free_kv_tokens)` prevents the solver from choosing a batch that cannot be allocated.
- **Preempt cooldown**: Preempted requests enter `_epreempt_cooldown` and are excluded from `_energy_build_views` until their TTFT SLO duration has elapsed. This prevents preempt→re-admit→preempt thrashing.
- **Single `allocate_slots` per request**: No double KV check.
- **GPU frequency set AFTER batch confirmed**: Not before.

**Helper methods**:
- `_energy_build_views(now_ms)`: Converts vLLM requests to `ReqView`. Sets `is_waiting=True` for waiting requests. Skips requests in `_epreempt_cooldown`. Handles partial-prefill (chunked) running requests.
- `_energy_get_arrival(req, now_ms)`: Priority: `send_time` from extra_args > `req.arrival_time` > `now_ms`
- `_energy_extract_slos(req)`: Pulls `(ttft_slo_ms, tpot_slo_ms, w_n)` from request's extra_args.
- `_energy_ensure_req_state(rid, req, now_ms)`: Initialises per-request state for online weight update on first sight.
- `_energy_update_weights(now_ms)`: Online adaptive update of `w_TTFT`/`w_TPOT` based on observed SLO performance.
- `_energy_release_preempt_cooldowns(now_ms)`: Releases requests from cooldown whose TTFT SLO duration has elapsed since preemption.

#### 3.9.3 Online Adaptive Weight Update

Implemented in `_energy_update_weights()`, runs at the start of every `schedule()` call:

- **TTFT update** (on first output token): `w_TTFT ← [w_TTFT + η_TTFT · w_n · (TTFT_obs/TTFT_slo − 1)]⁺`
- **TPOT update** (on request completion): `w_TPOT ← [w_TPOT + η_TPOT · w_n · (avg_TPOT_obs/TPOT_slo − 1)]⁺`

---

### 3.10 `vllm_patches/apply_patch.sh` (44 lines) — Patch installer

1. Clears Python bytecode cache in `vllm/energy_sched/`
2. Copies `solver.py`, `energy_model.py`, `frequency_controller.py`, `__init__.py` into `vllm/energy_sched/`
3. Removes old monkey-patch hook from `vllm/__init__.py` if present
4. Applies `scheduler_energy.patch` via `git apply`

### 3.11 `vllm_patches/unapply_patch.sh` (40 lines) — Patch rollback

1. Reverses `scheduler_energy.patch` via `git apply -R` (falls back to `git checkout` on failure)
2. Removes `vllm/energy_sched/` directory
3. Removes old monkey-patch hook if still present

### 3.12 `vllm_patches/scheduler_energy.patch` — Git patch for scheduler.py

Generated via `git diff` from the vLLM repo. Adds the `_energy_enabled` init block and 8 energy methods to `Scheduler`. Also adds `prev_step_scheduled_req_ids.discard()` to `_preempt_request` to prevent stale request IDs from persisting after preemption.

### 3.13 `vllm_patches/energy_scheduler.py` (1155 lines) — Legacy subclass implementation

An alternative subclass-based implementation (`EnergyScheduler(Scheduler)`) with `_materialise_batch()`, `_enforce_active_cap()`, `_kv_evict()`, etc. **Not used by `apply_patch.sh`** — not copied to the vLLM tree. Kept as a reference for the three-phase preemption approach (active-cap preemption → KV eviction → materialise-batch fallback) which differs from the embedded patch's admission-only design.

---

## 4. vLLM Edits

**File**: `vllm/v1/core/sched/scheduler.py` (modified via `scheduler_energy.patch`)

**Changes**:
1. `__init__`: Adds `self._energy_enabled` flag and energy scheduler init (solver, frequency controller, latency/power models, preempt cooldown dict) — gated by `VLLM_ENERGY_SCHEDULER=1` env var.
2. `schedule()`: Adds `if self._energy_enabled: return self._schedule_energy()` dispatch at the top — default path is completely unchanged.
3. `_preempt_request()`: Adds `self.prev_step_scheduled_req_ids.discard(request.request_id)` to prevent stale IDs after preemption.
4. 8 new methods: `_schedule_energy`, `_energy_release_preempt_cooldowns`, `_energy_build_views`, `_energy_get_arrival`, `_energy_extract_slos`, `_energy_ensure_req_state`, `_energy_update_weights`, (no `_energy_pick_victim`).

No changes to `vllm/__init__.py` (old monkey-patch hook removed).

## 5. Dataset Provenance

- **Repo**: `RyokoAI/ShareGPT52K` on Hugging Face
- **Auto-download**: `prepare_dataset.py` automatically downloads the dataset via `huggingface_hub` if the directory is missing, or re-downloads if Git LFS pointers are detected.
- **Trace**: 400 requests, first human message per conversation, prompt length 512–8000 chars
- **SLO parameters**: TTFT μ=4000ms σ=800ms, TPOT μ=100ms σ=40ms (truncated normal)
- **Arrival rate**: 4 req/s (uniform) — note: `main.sh` overrides this to 2.0 QPS at runtime

## 6. Results (latest run)

Parameters: `BETA=3.0, W_TTFT=2000.0 (initial), W_TPOT=1.0 (initial), SOLUTION_MODE=3 (H4), DEFAULT_MAX_NUM_SEQS=256, CUSTOM_MAX_NUM_SEQS=400, NUM_REQUESTS=400, RATE_QPS=2.0, FREQ_STRIDE=3, MAX_BATCH_SIZE=256, IS_CHUNKED_PREFILL=1, SOLVER_LMAX=8192, VLLM_MAX_BATCHED_TOKENS=8192, GPU_MEM_UTIL=0.95, MIN/MAX_OUT_TOK=1024/1024`.

| Metric | Default | Custom |
|--------|---------|--------|
| mean_ttft_ms | 132.01 | 162.99 |
| mean_tpot_ms | 42.24 | 54.46 |
| mean_ttft_violation_ms | 0.0 | 0.0 |
| mean_tpot_violation_ms | 1.07 | 2.30 |
| mean_normalized_ttft_violation | 0.0 | 0.0 |
| mean_normalized_tpot_violation | 0.078942 | 0.129457 |
| ttft_slo_attainment | 1.0 | 1.0 |
| tpot_slo_attainment | 0.93 | 0.865 |
| mean_power_w | 356.36 | 259.91 |
| total_energy_j | 82401.81 | 62385.42 |
| mean_solve_exec_ratio | 0.0 | 0.048616 |

**Notes on results**:
- All 400 requests completed in both modes (1024 output tokens each).
- **TTFT**: Both schedulers achieve 100% TTFT SLO attainment. Custom scheduler has slightly higher mean TTFT (163ms vs 132ms) but well within SLO targets.
- **TPOT trade-off**: Mean TPOT increases from 42.24ms to 54.46ms. TPOT SLO attainment drops from 93% to 86.5%. The normalized TPOT violation increases from 0.079 to 0.129, indicating moderate SLO degradation.
- **Energy saving**: Mean power drops from 356.36W to 259.91W (−27.1%), and total energy drops from 82.4kJ to 62.4kJ (−24.3%). With `β=3.0`, the solver aggressively selects lower GPU frequencies when the energy saving outweighs the SLO penalty.
- **Solver overhead**: `mean_solve_exec_ratio = 0.049` means the solver takes ~4.9% of batch execution time — acceptable overhead for the H4 heuristic.

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
5. Compares results → `results/demo/comparison.csv`

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
- `SOLUTION_MODE`: `1` (H2), `2` (H3), or `3` (H4, default)
- `NUM_REQUESTS`, `RATE_QPS`: workload size and arrival rate
- `VLLM_MAX_BATCHED_TOKENS`, `SOLVER_LMAX`: token budget controls
- See `main.sh` for the full list of knobs

To change trace generation parameters (SLO distributions, prompt filters), edit `scripts/prepare_dataset.py` and delete `trace.jsonl` to regenerate.

### Project Structure

```
energy_efficient_LLM_scheduling/
├── main.sh                          # Master experiment runner
├── trace.jsonl                      # Generated workload trace
├── experiment.md                    # This file
├── scripts/
│   ├── prepare_dataset.py           # Trace generator (ShareGPT → trace.jsonl)
│   ├── workload_sender.py           # Async HTTP workload replay
│   ├── power_logger.py              # GPU power sampling (pynvml → CSV)
│   ├── metrics_collector.py         # Results aggregation → summary.json
│   └── compare_results.py           # Side-by-side comparison → CSV
├── vllm_patches/
│   ├── solver.py                    # Pure algorithm (EnergySchedConfig, Alt1HeuristicSolver: H2+H3+H4)
│   ├── energy_model.py              # Latency + power models (LatencyParams, PowerParams)
│   ├── frequency_controller.py      # GPU SM clock control (pynvml wrapper)
│   ├── __init__.py                  # Package re-exports
│   ├── scheduler_energy.patch       # Git patch for vLLM scheduler.py
│   ├── apply_patch.sh               # Patch installer
│   ├── unapply_patch.sh             # Patch rollback
│   └── energy_scheduler.py          # Legacy subclass impl (reference only, not used)
└── results/
    └── demo/                        # Output directory (per TAG)
        ├── server_{default,custom}.log
        ├── power_{default,custom}.csv
        ├── results_{default,custom}.jsonl
        ├── summary_{default,custom}.json
        ├── iter_custom.log
        └── comparison.csv
```
