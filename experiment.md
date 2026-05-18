# Experiment Record — Energy-Efficient LLM Scheduling on vLLM

## 1. Problem Statement

We run a single vLLM server serving Qwen3-14B on an A800-SXM4-80GB GPU and compare two schedulers on the same workload:

- **Baseline**: vLLM's default FCFS scheduler, GPU clocks not locked.
- **Ours (custom)**: an energy-aware scheduler based on the **Alt-1 Heuristic** formulation — a three-step algorithm (priority scoring → greedy fill → joint prefix×frequency enumeration) with online adaptive weight updates. The scheduler selects both the GPU SM frequency and batch composition per iteration, locking the SM clock via `pynvml`.

Reported metrics include mean TTFT/TPOT, SLO violations, power, energy, and the mean solve-to-execution ratio.

## 2. Commands Executed

```bash
# Environment
conda activate myvllm

# Run experiment (dataset auto-download + patch + trace + experiment)
bash main.sh
```

`main.sh` handles everything end-to-end: applying the vLLM patch, verifying/downloading the dataset, generating the trace, running experiments, collecting metrics, and comparing results.

## 3. Files Created — Full Code Review

### 3.1 `main.sh` (284 lines) — Master experiment orchestrator

**Purpose**: Controls the full experiment lifecycle — applies the vLLM patch, generates the workload trace, launches the vLLM server, replays the workload, logs power, collects metrics, and compares results.

**L1–110: USER KNOBS block** — All tunable parameters are declared as Bash variables at the top of the file:

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
| `MAX_NUM_SEQS` | `64` | Max concurrent requests in the engine |
| `GPU_MEM_UTIL` | `0.90` | Fraction of GPU memory for KV cache |
| `NUM_REQUESTS` | `400` | Number of requests in the workload |
| `RATE_QPS` | `4` | Arrival rate (requests/second) |
| `MIN_OUT_TOK` / `MAX_OUT_TOK` | `64` / `1024` | Output token range per request |
| `TRACE_SEED` | `42` | Random seed for trace generation |
| `BETA` | `0.5` | Energy-utility trade-off (larger = more energy-saving) |
| `W_TTFT` | `1000.0` | Initial weight for TTFT in priority calculation (mutable — drifts online) |
| `W_TPOT` | `1.0` | Initial weight for TPOT in priority calculation (mutable — drifts online) |
| `ETA_MS` | `200` | Per-iteration time budget η (ms); accepted for compat but currently unused by solver |
| `LMAX` | `0` | Max tokens per batch (0 = inherit vLLM default) |
| `FREQ_STRIDE` | `3` | Stride for frequency candidate subsampling |
| `EVICTION_MODE` | `2` | KV cache eviction strategy (1=conservative, 2=incremental, 3=preempt) |
| `SOLUTION_MODE` | `2` | Solver heuristic (1=H2 freq-indep priority, 2=H3 freq-dep priority) |
| `POWER_INTERVAL_S` | `0.1` | GPU power sampling interval (seconds) |

**L113**: Captures the script directory so all paths are absolute regardless of CWD.

**L116–126**: Conda activation. Tries four possible `conda.sh` locations (miniconda3, anaconda3, /opt/conda), sources the first one found, then activates the `myvllm` environment.

**L131**: Calls `apply_patch.sh` to copy the energy scheduler Python files into the vLLM source tree. This is idempotent — it checks for a sentinel marker before appending.

**L134–141**: Conditional trace generation. If `trace.jsonl` already exists, it is reused. Delete the file to force regeneration.

**L145–265: `run_experiment()` function** — The core experiment runner:

- **L148–152**: Maps `"default"`/`"custom"` label to the output file suffix.
- **L156–174**: Builds the server environment variable array. For baseline, `VLLM_ENERGY_SCHEDULER=0`. For custom mode, sets `VLLM_ENERGY_SCHEDULER=1` plus all hyperparameters (`VLLM_ENERGY_BETA`, `VLLM_ENERGY_W_TTFT`, `VLLM_ENERGY_W_TPOT`, `VLLM_ENERGY_LMAX`, `VLLM_ENERGY_FREQ_STRIDE`, `VLLM_ENERGY_EVICTION_MODE`, `VLLM_ENERGY_SOLUTION_MODE`, `VLLM_ENERGY_ETA_MS`, `VLLM_ENERGY_GPU_INDEX`, `VLLM_ENERGY_ITER_LOG`).
- **L182–195**: Launches the vLLM server as a background process. Key flags:
  - `--enforce-eager`: Disables CUDA graphs (needed because frequency changes invalidate graph caches)
  - `--no-async-scheduling`: Disables async scheduling so the scheduler sees all running/waiting requests at each iteration
  - `--no-enable-chunked-prefill`: Disables chunked prefill (our scheduler operates at the batch level)
  - `--no-enable-prefix-caching`: Disables prefix caching (simplifies KV cache accounting)
  - `--enable-logging-iteration-details`: Enables detailed per-iteration logging in server log
  - stdout/stderr redirected to `server_${label}.log`
- **L198–211**: Health check loop. Polls `http://localhost:PORT/health` every 2 seconds for up to 240 seconds. If the server process dies before becoming ready, prints the log and exits.
- **L214–219**: Starts `power_logger.py` as a background process, sampling GPU power every 0.1s.
- **L222–227**: Runs `workload_sender.py` synchronously — it blocks until all requests are done.
- **L230–232**: Stops the power logger via `kill`.
- **L235–237**: Stops the vLLM server via `kill`.
- **L240–248**: Resets GPU clocks to default after custom mode. Uses the `FrequencyController` Python class first, falls back to `nvidia-smi -rgc / -rmc`.
- **L251–263**: Runs `metrics_collector.py` to aggregate results into `summary_${label}.json`.

**L268–274**: Sequential experiment execution. If `MODE` is `"default"` or `"both"`, runs baseline. If `"custom"` or `"both"`, runs custom.

**L277–284**: Runs `compare_results.py` to produce a side-by-side comparison table and CSV.

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

**The fix**: The workload sender now records a `send_time` (wall-clock epoch in seconds) *before* the HTTP POST and passes it to the server via `vllm_xargs.send_time`. The energy scheduler's `_get_arrival_ms()` method (see §3.9) uses `send_time` as the authoritative arrival time, falling back to `req.arrival_time` only if `send_time` is absent. TTFT measurement in the sender also uses `send_time` as the baseline. This captures the full end-to-end latency including HTTP and engine-queue delays.

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

### 3.5 `scripts/metrics_collector.py` (173 lines) — Metrics aggregation

**Purpose**: Reads `results.jsonl` and `power.csv` from a completed experiment, computes summary statistics, and writes `summary.json`.

**L10–16: `trapz()`**: Trapezoidal integration function. Given arrays of time (seconds) and power (watts), computes total energy in joules.

**L19–38: `interpolate_power()`**: Linear interpolation of the power trace at an arbitrary timestamp. Used to compute exact power values at the boundaries of the active window.

**L41–61: `windowed_energy()`**: Computes energy and mean power over a specific time window `[start_ts, end_ts]`:
- Interpolates power at window boundaries via `interpolate_power()`.
- Extracts power samples that fall within the window.
- Integrates energy over this windowed sub-trace via `trapz()`.

This is a significant improvement over the previous approach that integrated over the entire power log duration. The window is defined as `[first_send_time, last_complete_time]` — the active period when requests were actually being processed. This eliminates idle-period power from inflating the energy measurement.

**L64–76: `solve_exec_ratio()`**: Reads the iteration log and returns mean of `solve_ms / exec_ms`.

**L79–169: `main()`**:
- Filters completed requests (HTTP 200, no error).
- Computes mean TTFT, mean TPOT, SLO violations, and attainment rates.
- **L127–134**: Extracts `first_send = min(send_time)` and `last_complete = max(complete_time)` across all requests — these define the active window for energy computation.
- **L136–145**: Calls `windowed_energy()` to compute energy only over the active period.
- Writes `summary.json` with all metrics.

---

### 3.6 `scripts/compare_results.py` (62 lines) — Result comparison

**Purpose**: Reads two `summary.json` files (default and custom), prints a side-by-side table, and writes `comparison.csv`.

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

### 3.9 `vllm_patches/energy_scheduler.py` (963 lines) — Core energy scheduler

**Purpose**: The main scheduling algorithm. Replaces vLLM's default scheduler with an energy-aware one that jointly selects GPU frequency and batch composition per iteration, with online adaptive SLO-weight updates.

#### Part (a): Mathematical Formulation — Alt-1 Heuristic

The scheduler implements the **Alt-1 Heuristic** — a cheap approximation to the Alt-1 formulation (soft exponential deadline penalty). Instead of the exact τ-breakpoint enumeration (which is `O(|T|·|F|·N log N)`), this heuristic runs in `O(N log N + |F|·|B̂|²)` per solve().

##### Notation and Conventions

All time-related arithmetic inside the solver is in **SECONDS**. At the boundary with `energy_model` (which returns ms) and `ReqView` (which carries `deadline_ms` / `wait_ms` for backward compatibility with main.sh configs), there is exactly one `/1000` conversion at ingest and one `×1000` conversion on the `et_pred` returned to the caller.

Key quantities per request `n` at iteration `i`:

| Symbol | Definition | Units |
|---|---|---|
| `r_n` | `w_n · w_TTFT` (prefill) or `w_n · w_TPOT` (decode) — baseline reward | dimensionless |
| `s_n` | `deadline_n − T_{i,n}` — slack (positive = on time, negative = overdue) | seconds |
| `ℓ_{i,n}` | per-iteration token cost: `num_prompt_tokens` for prefill, `1` for decode | tokens |
| `T_{i,n}` | time waited since arrival (for prefill) or since last execution (for decode) | seconds |
| `ET_i(B, f)` | predicted batch execution time at frequency `f` | seconds |
| `P(f)` | GPU power draw at frequency `f` | watts |

##### Step 1: One-Shot Priority

Each request is scored once:

```
q_n = r_n · min(exp(−s_n), CAP) / ℓ_{i,n}
```

where:
- `exp(−s_n)` is a smooth urgency proxy:
  - `s_n` large positive → comfortably before deadline → low urgency → `q_n` small
  - `s_n ≈ 0` → near deadline → `exp(−s_n) ≈ 1`
  - `s_n` large negative → deeply overdue → urgency saturates at `CAP`
- `CAP = 200000.0` — caps the boost of deeply-overdue requests so a single item does not arbitrarily dominate the priority order (increased from 20000.0 to give overdue requests a stronger priority signal before saturation)
- The `/ℓ_{i,n}` converts `r_n · urgency` into a "value-density per token" — suitable for a token-bounded knapsack greedy

##### Step 2: Density-Greedy Fill

Sort all requests by `q_n` descending. Admit in order while:
- `cum_tokens + ℓ_n ≤ L_max` (token budget)
- `|B| < B_max` (batch size limit)

Requests that would exceed `L_max` are **skipped** (not stopped) — smaller items further down the order may still fit. This produces the greedy batch `B̂`.

##### Step 3: Joint (Prefix, Frequency) Enumeration

Define nested prefixes of the greedy batch:

```
B_j ≜ {first j items of B̂ in admission order},   j = 1, ..., |B̂|
```

These form a nested chain: `B_1 ⊂ B_2 ⊂ ... ⊂ B_{|B̂|}`.

Jointly enumerate `(B_j, f) ∈ {B_1, ..., B_{|B̂|}} × F` and pick:

```
(j*, f*) = argmax_{j,f}  Σ_{n∈B_j} r_n · exp(−[ET_i(B_j, f) − s_n]₊) − β · P(f) · ET_i(B_j, f)
```

where `[x]₊ = max(x, 0)`.

The objective has two terms:
1. **Utility**: `Σ r_n · exp(−[ET_i − s_n]₊)` — total reward scaled by a soft deadline penalty. When `ET_i ≤ s_n` (batch completes before the request's deadline), the exponential is `exp(0) = 1` (full reward). When `ET_i > s_n`, the reward decays exponentially with the overshoot.
2. **Energy cost**: `β · P(f) · ET_i(B_j, f)` — energy consumed in joules, scaled by β. Higher β penalises energy consumption more aggressively.

**Note**: The `CAP` only applies to the priority `q_n` in Step 2 (for the admission order). The objective in Step 3 uses the **uncapped** exponential form from the original Alt-1 utility.

**Plan A — Always Commit**: The solver initialises `best_J = −∞` and always picks the argmax `(j*, f*)` regardless of its sign. Rationale: vLLM will execute *something* every iteration anyway (continuous batching), so picking the least-bad `(B_j, f)` strictly dominates falling back to the default scheduler.

##### Incremental Optimisation (Vectorised)

Since `{B_j}` is nested, computing `ET_i(B_j, f)` for all `j` is essentially free:
- A single `np.cumsum` over per-request workload contributions yields `num_p[j]` and `num_d[j]` for every prefix.
- Mode indicators `I_p(B_j)`, `I_d(B_j)` are monotone (0→1, never reset), computed by `cumsum(...) > 0`.
- The full Step 3 vectorises to a `(|F|, |B̂|, |B̂|)` matrix product plus one argmax.

Batch execution time at prefix `j` and frequency `f`:

```
ET_i(B_j, f) = (num_p[j] / f + num_d[j] / f^α + t_c) / 1000
```

where:
- `num_p[j] = Σ_{n∈B_j, prefill} (a_p·l_q² + b_p·l_q·l_kv + c_p·l_q) + I_p · w_pf`
- `num_d[j] = Σ_{n∈B_j, decode} (a_d·l_kv + b_d) + I_d · w_dec`

##### Two Heuristic Modes (controlled by `SOLUTION_MODE`)

**Heuristic 2 (H2, `solution_mode=1`)**: `_solve_h2()`
- Frequency-**independent** priority: computes `q_n` once using `exp(−s_n)`, applies the same admission order for all frequencies.
- Steps: one greedy fill → joint `(B_j, f)` enumeration over prefixes of that single `B̂`.
- Complexity: `O(N log N + |F|·|B̂|²)`

**Heuristic 3 (H3, `solution_mode=2`)**: `_solve_h3()` — **current default**
- Frequency-**dependent** priority: the priority score incorporates the per-request energy cost at each candidate frequency:

```
q_n(f) = [r_n · min(exp(−s_n), CAP) − β · P(f) · t_n(f)] / ℓ_n
```

  where `t_n(f)` is the per-request time contribution at frequency `f`.

- For each frequency `f`, computes a separate admission order, a separate greedy batch `B̂(f)`, and a separate prefix search. Picks the global argmax `(j*, f*)` across all frequencies.
- This is more expensive but produces better solutions when energy cost varies significantly across frequencies.
- **Exp decomposition** optimisation: `exp(−max(ET_j − s_n, 0))` is decomposed as `min(exp(s_n) · exp(−ET_j), 1)`, avoiding recomputation of `exp(s_n)` across prefixes.
- Complexity: `O(|F| · (N log N + |B̂|²))`

#### Part (b): Online Adaptive Weight Update ("Adaptive Control")

The weights `w_TTFT` and `w_TPOT` are **not fixed** — they are updated online based on observed SLO performance. This is implemented in `_online_update_weights()`, which runs at the beginning of every `schedule()` call (before the solver), so the fresh weights are picked up immediately.

**TTFT update** — fires upon observing the first output token of request `n`:
```
w_TTFT ← [w_TTFT + η_TTFT · w_n · (TTFT_obs / TTFT_slo − 1)]⁺
```

**TPOT update** — fires upon completion of request `n` (request leaves the scheduler's visibility):
```
w_TPOT ← [w_TPOT + η_TPOT · w_n · (avg_TPOT_obs / TPOT_slo − 1)]⁺
```

where:
- `[x]⁺ = max(0, x)` (non-negativity projection)
- `η_TTFT = 1.0`, `η_TPOT = 1.0` (hardcoded learning rates)
- `TTFT_obs = now_ms − arrival_ms` (observed TTFT for request `n`)
- `avg_TPOT_obs = (now_ms − first_tok_ms) / (num_output_tokens − 1)`
- `TTFT_slo`, `TPOT_slo` are per-request SLO targets from `extra_args`
- Multiple events in the same iteration are applied **serially** (each subsequent update sees the already-updated weight)

**Intuition**: When TTFT is consistently violated (`TTFT_obs/TTFT_slo > 1`), `w_TTFT` increases, making the solver prioritise prefill requests more aggressively. When TTFT SLOs are met (`ratio < 1`), `w_TTFT` decreases, allowing the solver to shift priority toward energy savings or decode throughput.

Per-request state for the online update is tracked in `self._req_state` (keyed by `request_id`), with fields: `arrival_ms`, `w_n`, `ttft_slo_ms`, `tpot_slo_ms`, `ttft_fired`, `first_tok_ms`, `last_num_out`.

#### Part (c): `send_time` as Authoritative Arrival Time

The method `_get_arrival_ms(req, now_ms)` determines the arrival time for each request with the following priority:

1. **`send_time`** from `extra_args` (client-side wall-clock epoch) — preferred
2. `req.arrival_time` (vLLM internal timestamp) — fallback
3. `now_ms` — last resort

This is crucial because vLLM's `req.arrival_time` is set inside `input_processor.process_inputs`, which can be delayed when the engine event loop is busy under high load. The client-side `send_time` captures the full queuing delay.

#### Part (d): KV Cache Eviction (3 modes)

`_kv_evict()` ensures the chosen batch fits in the KV cache:

- **Mode 1 (conservative)**: Each request's full KV size in blocks is counted against free blocks. If insufficient, the request with lowest adjusted utility is removed from the batch. Repeats until the batch fits.
- **Mode 2 (incremental)**: Only new blocks needed this iteration are counted (matches vLLM's `allocate_slots` logic). Otherwise same shrink policy as mode 1.
- **Mode 3 (preempt)**: Preempts non-chosen running requests to free their KV blocks (resets them to prefill state). Falls back to mode 1/2 shrinking if still insufficient.

#### Part (e): vLLM Integration

**`make_energy_scheduler_class()`**: Factory function that creates an `EnergyScheduler` class subclassing vLLM's `Scheduler`.

**`__init__`**:
- Loads config from environment variables.
- Loads latency and power model parameters.
- Creates the frequency controller singleton.
- Builds frequency candidates list (from controller, appending 1410 if absent).
- Creates the `Alt1HeuristicSolver` instance.
- Sets `Lmax` from vLLM's `scheduler_config.max_num_batched_tokens`.
- Opens the iteration log file.
- Initialises `_req_state` dict for the online weight update.

**`_build_request_views(now_ms)`**: Converts vLLM's internal request objects into `ReqView` dataclasses:
- **Waiting (prefill)**: `wait_ms = now_ms − arrival_ms` (using `_get_arrival_ms`), `l_q = num_prompt_tokens`, `l_kv = 0`, `deadline_ms = ttft_ms`.
- **Running (decode)**: `wait_ms = now_ms − last_exec_ms` (time since this request was last executed), `l_q = 1`, `l_kv = num_computed_tokens`, `deadline_ms = tpot_ms`. Falls back to `arrival_ms` if no `last_exec_ms` is recorded.

**`_materialise_batch(chosen)`**: Temporarily hides unchosen requests from `self.waiting`/`self.running`, calls `super().schedule()`, then restores them.

**`schedule()`**: Main entry point, called by vLLM on every scheduling iteration:
1. Measures `exec_ms` — wall-clock gap since last `schedule()` exit.
2. Runs `_online_update_weights(now_ms)` — updates `w_TTFT`/`w_TPOT`.
3. Builds request views; captures `len_waiting_before` / `len_running_before` queue lengths.
4. Calls `solver.solve()`, sets GPU frequency. Records solver-chosen request IDs split by prefill/decode.
5. Runs KV eviction, materialises batch via `super().schedule()`.
6. Records `last_exec[req_id] = now_ms` for all chosen requests.
7. **Solver-vs-actual diagnostics**: Compares the solver's chosen batch against what `super().schedule()` actually scheduled. Computes `dropped_prefill`, `dropped_decode` (solver picked but parent didn't schedule), `extra_prefill`, `extra_decode` (parent scheduled but solver didn't pick), and `parent_preempted`. Prints a `[energy_sched-MISMATCH]` warning when the parent silently drops solver picks.
8. Logs iteration data with extended fields: `solve_ms, batch_size, n_prefill, n_decode, n_preempted, f_star, et_pred_ms, w_ttft, w_tpot, ttft_updates, tpot_updates, len_waiting_before, len_running_before, chosen_total_tokens, actual_prefill, actual_decode, actual_total_tokens, dropped_prefill, dropped_decode, extra_prefill, extra_decode, parent_preempted`.

---

### 3.10 `vllm_patches/apply_patch.sh` (36 lines) — Patch installer

Copies 4 files into `vllm/energy_sched/` and appends a sentinel-guarded hook to `vllm/__init__.py`.

### 3.11 `vllm_patches/unapply_patch.sh` (27 lines) — Patch remover

Removes the `vllm/energy_sched/` directory and strips the hook from `vllm/__init__.py`.

### 3.12 `vllm_patches/__init__.py` (1 line) — Package marker

---

## 4. vLLM Edits

**File**: `/home/ubuntu/lqs/vllm/vllm/__init__.py`

**Snippet** (appended, between `# <<< ENERGY_SCHED_HOOK >>>` marker and EOF):
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

**How the hook works**: When vLLM is imported, if `VLLM_ENERGY_SCHEDULER=1` is set, it monkey-patches `vllm.v1.core.sched.scheduler.Scheduler` with the energy-aware subclass. Pure no-op unless the env var is set.

## 5. Dataset Provenance

- **Repo**: `RyokoAI/ShareGPT52K` on Hugging Face
- **Auto-download**: `prepare_dataset.py` automatically downloads the dataset via `huggingface_hub` if the directory is missing, or re-downloads if Git LFS pointers are detected.
- **Trace**: 400 requests, first human message per conversation, prompt length 512–8000 chars
- **SLO parameters**: TTFT μ=4000ms σ=800ms, TPOT μ=100ms σ=40ms (truncated normal)
- **Arrival rate**: 4 req/s (uniform)

## 6. Results (latest run)

Parameters: `BETA=0.5, W_TTFT=1000.0 (initial), W_TPOT=1.0 (initial), SOLUTION_MODE=2 (H3), EVICTION_MODE=2, MAX_NUM_SEQS=64, NUM_REQUESTS=400, RATE_QPS=4, FREQ_STRIDE=3`.

| Metric | Default | Custom |
|--------|---------|--------|
| num_completed | 400 | 400 |
| num_failed | 0 | 0 |
| mean_ttft_ms | 22335.68 | 2237.17 |
| mean_tpot_ms | 53.64 | 123.16 |
| mean_ttft_violation_ms | 19655.05 | 57.44 |
| mean_tpot_violation_ms | 2.15 | 28.27 |
| ttft_slo_attainment | 0.3425 | 0.865 |
| tpot_slo_attainment | 0.8725 | 0.2375 |
| mean_power_w | 370.88 | 248.92 |
| total_energy_j | 71101.37 | 62415.83 |
| mean_solve_exec_ratio | 0.0 | 0.069486 |

**Notes on results**:
- All 400 requests completed in both modes.
- **TTFT improvement**: Custom scheduler reduces mean TTFT by 90.0% (22336ms → 2237ms) and TTFT SLO attainment improves from 34.25% to 86.5%. The combination of a lower initial `w_TTFT=1000` (which lets the online adaptive update converge faster) and the higher `EXP_CAP=200000` (giving overdue requests stronger priority signal) drives dramatic prefill prioritisation.
- **TPOT trade-off**: Mean TPOT increases from 53.64ms to 123.16ms, and TPOT SLO attainment drops from 87.25% to 23.75%. With `β=0.5`, the solver aggressively lowers GPU frequency to save energy, which lengthens decode latency. The energy savings come at the cost of TPOT performance.
- **Energy saving**: With `β=0.5`, the energy term is active — mean power drops from 370.88W to 248.92W (−32.9%), and total energy drops from 71.1kJ to 62.4kJ (−12.2%). The solver actively selects lower GPU frequencies when the energy saving outweighs the SLO penalty.
- **Solver overhead**: `mean_solve_exec_ratio = 0.069` means the solver takes ~6.9% of batch execution time — still acceptable, though higher than the previous `β=0.0` run due to the frequency-dependent H3 heuristic evaluating more candidates at non-trivial β.

## 7. How to Reproduce

```bash
cd /home/ubuntu/lqs/energy_efficient_LLM_scheduling
conda activate myvllm

# Run experiment (dataset is auto-downloaded if needed)
bash main.sh

# View results
cat results/demo/comparison.csv
```

To change experiment parameters, edit the USER KNOBS block at the top of `main.sh`. To change trace generation parameters (or force re-download of the dataset), edit the USER KNOBS block at the top of `scripts/prepare_dataset.py` and delete `trace.jsonl` to regenerate.
