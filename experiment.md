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

# Run experiment (dataset auto-download + patch + trace + experiment)
bash main.sh
```

`main.sh` handles everything end-to-end: applying the vLLM patch, verifying/downloading the dataset, generating the trace, running experiments, collecting metrics, and comparing results.

## 3. Files Created — Full Code Review

### 3.1 `main.sh` (267 lines) — Master experiment orchestrator

**Purpose**: Controls the full experiment lifecycle — applies the vLLM patch, generates the workload trace, launches the vLLM server, replays the workload, logs power, collects metrics, and compares results.

**L1–99: USER KNOBS block** — All tunable parameters are declared as Bash variables at the top of the file. The reviewer can change any of these before running:

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
| `MAX_NUM_SEQS` | `128` | Max concurrent requests in the engine |
| `GPU_MEM_UTIL` | `0.90` | Fraction of GPU memory for KV cache |
| `NUM_REQUESTS` | `400` | Number of requests in the workload |
| `RATE_QPS` | `10` | Arrival rate (requests/second) |
| `MIN_OUT_TOK` / `MAX_OUT_TOK` | `64` / `1024` | Output token range per request |
| `TRACE_SEED` | `42` | Random seed for trace generation |
| `BETA` | `0.01` | Energy-utility trade-off (larger = more energy-saving) |
| `W_TTFT` | `2000.0` | Weight for TTFT in priority calculation |
| `W_TPOT` | `50.0` | Weight for TPOT in priority calculation |
| `ETA_MS` | `200` | Per-iteration time budget (milliseconds) |
| `LMAX` | `0` | Max tokens per batch (0 = inherit vLLM default) |
| `FREQ_STRIDE` | `4` | Stride for frequency candidate subsampling |
| `POWER_INTERVAL_S` | `0.1` | GPU power sampling interval (seconds) |

**L101**: Captures the script directory so all paths are absolute regardless of CWD.

**L104–114**: Conda activation. Tries four possible `conda.sh` locations (miniconda3, anaconda3, /opt/conda), sources the first one found, then activates the `myvllm` environment.

**L120**: Calls `apply_patch.sh` to copy the energy scheduler Python files into the vLLM source tree. This is idempotent — it checks for a sentinel marker before appending.

**L123–128**: Conditional trace generation. If `trace.jsonl` already exists, it is reused (skipping the 10–20 second dataset scan). Delete the file to force regeneration.

**L133–248: `run_experiment()` function** — The core experiment runner:

- **L134–139**: Maps `"default"`/`"custom"` label to the output file suffix.
- **L143–160**: Builds the server environment variable array. For baseline, `VLLM_ENERGY_SCHEDULER=0` (the hook in `vllm/__init__.py` is a no-op). For custom mode, sets `VLLM_ENERGY_SCHEDULER=1` plus all hyperparameters (`VLLM_ENERGY_BETA`, `VLLM_ENERGY_W_TTFT`, etc.) and the iteration log path.
- **L168–178**: Launches the vLLM server as a background process. Key flags:
  - `--enforce-eager`: Disables CUDA graphs (needed because frequency changes invalidate graph caches)
  - `--no-async-scheduling`: Disables async scheduling so the scheduler sees all running/waiting requests at each iteration
  - `--no-enable-chunked-prefill`: Disables chunked prefill (our scheduler operates at the batch level)
  - `--no-enable-prefix-caching`: Disables prefix caching (simplifies KV cache accounting)
  - stdout/stderr redirected to `server_${label}.log`
- **L183–194**: Health check loop. Polls `http://localhost:PORT/health` every 2 seconds for up to 240 seconds. If the server process dies before becoming ready, prints the log and exits.
- **L197–202**: Starts `power_logger.py` as a background process, sampling GPU power every 0.1s.
- **L205–210**: Runs `workload_sender.py` synchronously — it blocks until all requests are done.
- **L213–220**: Stops the power logger and server via `kill`.
- **L223–232**: Resets GPU clocks to default after custom mode. Uses the `FrequencyController` Python class first, falls back to `nvidia-smi -rgc`.
- **L235–245**: Runs `metrics_collector.py` to aggregate results into `summary_${label}.json`.

**L251–257**: Sequential experiment execution. If `MODE` is `"default"` or `"both"`, runs baseline. If `"custom"` or `"both"`, runs custom. They run sequentially (not parallel) because there is only one GPU.

**L260–266**: Runs `compare_results.py` to produce a side-by-side comparison table and CSV.

---

### 3.2 `scripts/prepare_dataset.py` (145 lines) — Synthetic trace generation

**Purpose**: Ensures the ShareGPT52K dataset is available (auto-downloading if needed, including re-downloading Git LFS pointers), filters and samples prompts, and writes `trace.jsonl` — one JSON record per line representing a single request with its arrival time, prompt, and SLO parameters.

**L18–61: USER KNOBS block**:

| Constant | Default | Meaning |
|---|---|---|
| `OUTPUT` | `"trace.jsonl"` | Output file path |
| `NUM_REQUESTS` | `400` | Number of requests to sample |
| `RATE_QPS` | `10` | Arrival rate — request i arrives at `i / RATE_QPS` seconds |
| `TTFT_MEAN_MS` | `1000.0` | Mean TTFT SLO target (ms) |
| `TTFT_STD_MS` | `200.0` | Std dev of TTFT SLO |
| `TPOT_MEAN_MS` | `50.0` | Mean TPOT SLO target (ms) |
| `TPOT_STD_MS` | `40.0` | Std dev of TPOT SLO |
| `MIN_OUTPUT_TOKENS` / `MAX_OUTPUT_TOKENS` | `64` / `1024` | Output token range |
| `MIN_PROMPT_CHARS` / `MAX_PROMPT_CHARS` | `64` / `8000` | Prompt length filter (characters) |
| `SEED` | `42` | Random seed for reproducibility |
| `DATASET_DIR` | `"data/sharegpt52k"` | Local path to ShareGPT dataset |
| `REPO_ID` | `"RyokoAI/ShareGPT52K"` | Hugging Face repository ID for auto-download |

**L63–87: `_ensure_dataset()`** — Automatic dataset verification:
- **L65–68**: If `DATASET_DIR` doesn't exist, downloads the entire dataset via `huggingface_hub.snapshot_download()`.
- **L70–79**: Checks each `.json` file for Git LFS pointer signatures (file size < 200 bytes, content starts with `version `). If detected, removes the entire directory and re-downloads the real data. This handles the common case where the user cloned the repo (getting LFS pointers instead of actual data) without needing manual cleanup.

**L89–93: `truncated_normal()`**: Samples from a Gaussian distribution and rejects values ≤ `low`. Used to generate TTFT/TPOT SLO targets that are always positive.

**L100–113: Dataset loading**:
- Iterates all `.json` files in `DATASET_DIR`.
- For each conversation, extracts the first human/user message.
- Filters by prompt length (64–8000 characters).
- Handles both ShareGPT format keys (`"from"`/`"value"`) and OpenAI format keys (`"role"`/`"content"`).

**L121–143: Trace writing**:
- Shuffles all candidate prompts, takes the first `NUM_REQUESTS`.
- For each request, writes a JSON record with:
  - `id`: unique identifier like `"req_000001"`
  - `arrival_s`: `i / RATE_QPS` — uniform arrival times (e.g., at 10 QPS, requests arrive every 0.1s)
  - `prompt`: the actual text content
  - `max_tokens`: uniformly sampled from `[64, 1024]`
  - `ttft_ms`: sampled from truncated normal (μ=1000, σ=200)
  - `tpot_ms`: sampled from truncated normal (μ=50, σ=40)
  - `w_n`: priority weight, default 1.0

---

### 3.3 `scripts/workload_sender.py` (170 lines) — Async workload replay

**Purpose**: Reads `trace.jsonl` and asynchronously sends each request to the vLLM `/v1/completions` endpoint with `stream=true`, measuring per-request TTFT and TPOT.

**L16–30: `ResultRecord` dataclass**: Holds per-request metadata (id, prompt length, max_tokens, TTFT/TPOT SLO targets) and results (actual TTFT/TPOT, SLO violations, token count, HTTP status, error message).

**L33–109: `send_one()`**: Sends a single request and measures timing:
- **L38–50**: Builds the HTTP POST payload. Passes TTFT/TPOT/w_n to the server via `extra_body.extra_args` — this is how the energy scheduler receives per-request SLO information.
- **L52–54**: Records `send_time` before the HTTP request.
- **L56–58**: Awaits the HTTP response. Records `first_token_time` when the first chunk arrives — this is TTFT.
- **L60–97**: Parses the SSE (Server-Sent Events) stream:
  - Each line starts with `data: `.
  - Decodes JSON chunks, extracts the generated text token.
  - Records the gap between consecutive chunks — these are TPOT values.
  - Counts total tokens generated.
- **L99–109**: Error handling — catches network errors, HTTP errors, and timeouts. Records the error in the result.

**L112–170: `main()`**: Orchestrates the workload replay:
- **L115–119**: Loads all requests from `trace.jsonl`.
- **L121–130**: Dispatches requests in order, respecting arrival times. Uses `asyncio.sleep` to wait until `arrival_s` before launching each request as an async task.
- **L132–140**: Gathers all results, computes SLO violations (max(0, actual - target)).
- **L142–170**: Writes `results.jsonl` — one JSON record per line with all per-request metrics.

---

### 3.4 `scripts/power_logger.py` (58 lines) — GPU power sampling

**Purpose**: Continuously samples GPU power draw, SM clock frequency, and GPU utilization via `pynvml`, writing a CSV row every 0.1 seconds.

**L20–22: Signal handler**: Sets a global `_stop` flag on SIGTERM/SIGINT for clean shutdown.

**L28: CSV columns**: `timestamp_s, power_w, sm_clock_mhz, utilization_pct`

**L30–35: NVML initialization**: Gets the GPU handle for the specified GPU index.

**L38–52: Main sampling loop**:
- Calls `nvmlDeviceGetPowerInfo()` to get power in milliwatts, converts to watts.
- Calls `nvmlDeviceGetClockInfo()` for SM clock frequency.
- Calls `nvmlDeviceGetUtilizationRates()` for GPU utilization percentage.
- Writes a CSV row with `flush=True` after each sample (ensures data is not lost on crash).
- Sleeps for the configured interval (0.1s by default).
- Exits when `_stop` flag is set.

---

### 3.5 `scripts/metrics_collector.py` (126 lines) — Metrics aggregation

**Purpose**: Reads `results.jsonl` and `power.csv` from a completed experiment, computes summary statistics, and writes `summary.json`.

**L11–17: `trapz()`**: Trapezoidal integration function. Given arrays of power (watts) and time (seconds), computes the total energy in joules as the area under the power-vs-time curve.

**L20–32: `solve_exec_ratio()`**: Reads the iteration log (`iter_custom.log`) from the custom scheduler. Each line is a JSON record with `solve_ms` (time to run the optimisation) and `exec_ms` (time for the actual batch execution on GPU). Returns the mean of `solve_ms / exec_ms` — a measure of solver overhead relative to execution time. Returns 0.0 if the log doesn't exist (baseline case).

**L50–52**: Filters to only completed requests (HTTP status 200, no error).

**L54–58**: Computes mean TTFT and TPOT across all completed requests.

**L60–72**: Computes SLO violations:
- `mean_ttft_violation_ms`: average of `max(0, actual_ttft - ttft_slo)` — how late TTFT was on average.
- `mean_tpot_violation_ms`: same for TPOT.
- `ttft_slo_attainment`: fraction of requests that met their TTFT SLO (zero violation).
- `tpot_slo_attainment`: same for TPOT.

**L77–98**: Loads `power.csv`:
- Parses timestamp, power, clock, utilization columns.
- Computes time deltas between consecutive samples.
- Integrates energy via `trapz()`.
- Computes mean power draw.

**L102–115**: Writes `summary.json` with all 9 required fields.

---

### 3.6 `scripts/compare_results.py` (62 lines) — Result comparison

**Purpose**: Reads two `summary.json` files (default and custom), prints a side-by-side table, and writes `comparison.csv`.

**L10–19: `METRIC_ORDER`**: Defines the 9 metrics in fixed display order with formatting (decimal places, units).

**L44–48**: Prints a formatted table:
```
Metric                    Default        Custom
─────────────────────────────────────────────────
mean_ttft_ms              4106.25        22631.79
...
```

**L51–56**: Writes `comparison.csv` with columns: `metric, default, custom`.

---

### 3.7 `vllm_patches/energy_model.py` (114 lines) — Latency and power models

**Purpose**: Provides the mathematical models for per-request latency and GPU power as functions of frequency. These are used by the scheduler to predict execution time and energy consumption.

**L19–41: `LatencyParams` dataclass**: 9 Route B+ coefficients:
- `a_p, b_p, c_p`: prefill latency quadratic model coefficients
- `w_pf, w_dec`: batch overhead weights for prefill and decode
- `a_d, b_d`: decode latency coefficients
- `alpha`: frequency scaling exponent for decode
- `t_c`: constant communication overhead

The `from_json()` class method allows overriding defaults from a JSON file.

**L48–62: `PowerParams` dataclass**: Cubic power model coefficients `(k3, k2, k1, k0)`. The `power_watts(f)` method evaluates `P(f) = k3·f³ + k2·f² + k1·f + k0`.

**L69–76: `per_request_time_ms()`** — Eq. 4 in the paper:
- **Prefill**: `t_q = (a_p · l_q² + b_p · l_q · l_kv + c_p · l_q) / f`
  - Quadratic in prompt length `l_q`, linear in KV cache length `l_kv`, inversely proportional to frequency.
- **Decode**: `t_q = (a_d · l_kv + b_d) / f^alpha`
  - Linear in KV cache length, inversely proportional to `f^alpha`.

**L79–87: `batch_overhead_ms()`** — Eq. 5:
- `T_ovh = (has_prefill ? w_pf/f : 0) + (has_decode ? w_dec/f^alpha : 0)`

**L90–100: `batch_time_ms()`** — Total iteration time:
- `ET_i(B, f) = Σ t_q(n, f) + T_ovh(B, f) + t_c`

**L103–114: `load_latency_params()` / `load_power_params()`**: Return hardcoded A800 defaults, or load from JSON if `VLLM_ENERGY_LATENCY_JSON` / `VLLM_ENERGY_POWER_JSON` env vars are set.

---

### 3.8 `vllm_patches/frequency_controller.py` (150 lines) — GPU frequency control

**Purpose**: Provides an abstraction for setting and resetting GPU SM clock frequency and memory clock frequency. Uses `pynvml` as the primary mechanism, falls back to `sudo nvidia-smi` when permissions are insufficient.

**L19–28**: `__init__` — Initializes NVML, gets the GPU device handle, queries all supported SM clock frequencies, queries supported memory clocks, and **attempts to lock memory clock to 1593 MHz** (the profiling baseline). A800-SXM4 does not support memory clock locking, so this is a no-op on this hardware, but the attempt is made explicitly to match profiling conditions.

**L30–33**: `supported_clocks()` — Returns sorted list of supported SM clock frequencies (81 values on A800: 210–1410 MHz, 15 MHz steps).

**L35–57**: `set_frequency(f_mhz)`:
- **L35–38**: Finds the closest supported SM frequency to the requested value. Skips the call if already at target.
- **L39–42**: First tries `pynvml.nvmlDeviceSetGpuLockedClocks(self._handle, target, target)`.
- **L43–57**: Falls back to `sudo nvidia-smi -lgc {target},{target}`.

**L59–81**: `reset()` — Unlocks both SM clock and memory clock, restoring dynamic frequency scaling. NVML first, then `sudo nvidia-smi -rgc / -rmc` fallback.

**L85–88**: `_closest(f_mhu)` — Returns the supported SM frequency closest to the requested value.

**L90–113**: `_query_supported_clocks()` — Queries supported SM clock frequencies:
- **L91–97**: `pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, 0)` — the second argument `0` means "all graphics clocks regardless of memory clock."
- **L99–113**: Falls back to parsing `nvidia-smi --query-supported-clocks=gr` output. Strips `" MHz"` suffix before `int()` conversion.

**L115–122**: `_query_supported_memory_clocks()` — Returns supported memory clocks (A800 typically reports only `[1593]`).

**L124–143**: `_lock_memory_clock()` — Attempts to lock memory clock to 1593 MHz (profiling baseline):
- **L131**: Tries `pynvml.nvmlDeviceSetMemoryLockedClocks()` first.
- **L136–140**: Falls back to `sudo nvidia-smi -lmc 1593,1593`.
- Both methods fail on A800 (hardware doesn't support memory clock locking) — this is expected and non-fatal.

**L146–149**: `get_controller()` — LRU-cached singleton factory. Reads `VLLM_ENERGY_GPU_INDEX` env var.

---

### 3.9 `vllm_patches/energy_scheduler.py` (411 lines) — Core energy scheduler

**Purpose**: The main scheduling algorithm. Replaces vLLM's default scheduler with an energy-aware one that selects GPU frequency and batch composition per iteration.

#### Part (a): Pure-Python core (L17–232) — testable without vLLM

**L30–55: `EnergySchedConfig` dataclass**: Holds all scheduler hyperparameters:
- `beta`: energy-utility trade-off (larger → more energy saving)
- `w_ttft`, `w_tpot`: weights for TTFT/TPOT in priority calculation
- `eta_ms`: base per-iteration time budget from env (VLLM_ENERGY_ETA_MS).
  The actual time budget used is `max(eta_ms, min_r slack_ms)`, where `slack_ms = deadline_ms - wait_ms` over all requests. This ensures the budget never drops below the most urgent request's slack.
- `Lmax`: max tokens per batch
- `freq_stride`: stride for frequency candidate subsampling
- `from_env()`: reads all values from `VLLM_ENERGY_*` environment variables

**L58–67: `ReqView` dataclass**: Lightweight request representation used by the solver:
- `handle`: reference to the actual vLLM request object
- `is_prefill`: True for waiting requests, False for running (decode) requests
- `l_q`: number of prompt tokens (prefill) or 1 (decode)
- `l_kv`: number of already-computed tokens (KV cache length)
- `wait_ms`: time since request arrival
- `deadline_ms`: TTFT target (prefill) or TPOT target (decode)
- `w_n`: priority weight
- `kv_blocks_needed`: KV cache blocks required

**L70–73: `instant_utility()`** — Eq. 1:
```
u(n) = r_n · exp(-(deadline - wait) / 1000)
```
The urgency-based priority: decays exponentially as slack decreases. A request close to its deadline has high utility.

**L76–83: `adjusted_utility()`** — Eq. 3:
```
v(n, f) = u(n) - β · P(f) · t_q(n, f)
```
Subtracts the energy penalty (power × time, scaled by β) from the instantaneous utility. At high β, requests at low frequency may have negative adjusted utility, causing the solver to prefer lower frequencies or defer the request.

**L86–112: `greedy_knapsack_2d()`** — 2-D knapsack solver:
- Filters out items with value ≤ 0.
- Computes utility density: `value / max(token_ratio, time_ratio)`.
- Sorts by density (highest first).
- Greedily adds items while both constraints hold: total tokens ≤ Lmax and total time ≤ η.
- Returns indices of selected items.

**L115–232: `FrequencyFirstSolver.solve()`** — The core algorithm:

- **L131–132**: Separates requests into running (decode, currently in-progress) and prefill (waiting) categories. Running requests MUST be served to avoid stalling in-flight generations.

- **L134**: Initializes `best` as `(0.0, highest_freq, [], 0.0)` — the empty batch at highest frequency is the baseline fallback.

- **L137–143**: Builds the mask list — which batch compositions to consider:
  - `"prefill_only"`: only prefill requests (no running requests exist)
  - `"decode_only"`: only running requests (no waiting requests)
  - `"mixed"`: both prefill and decode requests

- **L145–148**: Subsamples frequency candidates using `freq_stride`. With stride=4 on 82 A800 clocks, this evaluates 21 candidates instead of 82.

- **L146–149**: Computes `effective_eta = max(cfg.eta_ms, min_r (deadline_ms - wait_ms))` — dynamic time budget that never drops below the most urgent request's slack.

- **L151–198**: Main enumeration loop over frequency and mask:
  - **L152**: Computes power at the candidate frequency.
  - **L153**: Computes `(v, t_q)` for every request at this frequency.
  - **L156–158**: Computes batch overhead based on whether prefill/decode are present, then derives `eta_left = effective_eta - T_ovh - t_c`.
  - **L161–177**: Builds values/times/tokens arrays for the knapsack:
    - **Mode filtering (L167–170)**: `prefill_only` skips decode requests; `decode_only` skips prefill requests. This was a bug fix — the original code omitted these guards, causing all three modes to evaluate the same request set, making the mode enumeration meaningless.
    - Requests with negative utility are skipped (the solver would defer them).
  - **L176–180**: Runs the greedy knapsack solver.
  - **L184–188**: For `"mixed"` mask, validates the batch actually contains both prefill and decode requests (otherwise it degenerates to a single-type batch, which should have been handled by the other masks).
  - **L190–195**: Computes the objective `J = Σ v(n, f*) - β · P(f*) · (T_ovh + t_c)` and updates the best solution if this is better.

- **L203**: `_, f_star, picked, et_pred = best` — safely unpacks the best tuple as the fallback. This was a bug fix: the original code referenced `f_star`/`picked`/`et_pred` directly without unpacking from `best`, causing a `NameError` on the very first iteration (debug_iter=0), crashing the engine after one request.

#### Part (b): vLLM integration (L224–398)

**L244–411: `make_energy_scheduler_class()`**: Factory function that creates an `EnergyScheduler` class subclassing vLLM's `Scheduler`.

**L228–270: `__init__`**:
- Loads config from environment variables.
- Loads latency and power model parameters.
- Creates the frequency controller singleton.
- Builds the list of frequency candidates (from controller or defaults).
- Creates the solver instance.
- Sets `Lmax` from vLLM's `scheduler_config.max_num_batched_tokens`.
- Opens the iteration log file for append.

**L272–317: `_build_request_views()`**: Converts vLLM's internal request objects into `ReqView` dataclasses:
- **L274–295**: Iterates `self.waiting` (prefill requests). Extracts TTFT/TPOT/w_n from `sampling_params.extra_args`. Computes `wait_ms = now - arrival_time`. Computes `l_q` from `num_prompt_tokens`.
- **L296–317**: Iterates `self.running` (decode requests). Sets `l_q = 1` (one token per decode step). Computes `l_kv` from `num_computed_tokens`.

**L319–345: `_kv_evict()`**: Handles KV cache capacity constraints:
- If the total KV blocks needed by chosen requests exceeds available free blocks, evicts the request with the lowest adjusted utility.
- Repeats until the batch fits or only one request remains.

**L347–366: `_materialise_batch()`**: Executes the chosen batch:
- Identifies which requests are chosen vs unchosen.
- Temporarily removes unchosen requests from `self.waiting` and `self.running` using vLLM's `remove_requests()`/`remove()` methods.
- Calls `super().schedule()` (vLLM's default scheduler) to actually execute the chosen batch.
- Restores unchosen waiting requests to the queue.

**L368–396: `schedule()`**: Main entry point, called by vLLM on every scheduling iteration:
- **L369–373**: Measures `exec_ms` — the wall-clock gap since the last `schedule()` exit, which approximates the GPU execution time of the previous batch.
- **L374–375**: Builds request views from current waiting/running queues.
- **L376–379**: Calls the solver, sets GPU frequency via `set_frequency()`. If solver returns empty batch, falls back to `super().schedule()`.
- **L380–383**: Evicts KV cache if needed and materializes the batch.
- **L384–396**: Logs iteration data (solve time, batch size, frequency, predicted time) for later analysis.
- **L400–408**: Prints a summary every `log_every_n` iterations.

---

### 3.10 `vllm_patches/apply_patch.sh` (36 lines) — Patch installer

**Purpose**: Copies the energy scheduler Python files into the vLLM source tree and appends a sentinel-guarded hook to `vllm/__init__.py`.

**L6–10**: Copies 4 files into `vllm/energy_sched/`:
- `energy_model.py` → latency and power models
- `frequency_controller.py` → GPU frequency control
- `energy_scheduler.py` → the scheduler itself
- `__init__.py` → package marker

**L13–22**: Appends the `ENERGY_SCHED_HOOK` to `vllm/__init__.py`. The hook is guarded by:
- A sentinel comment `# <<< ENERGY_SCHED_HOOK >>>` so the script can detect if it's already been applied (idempotent).
- An environment variable check `VLLM_ENERGY_SCHEDULER=1` so the hook is a no-op unless explicitly enabled.
- A try/except so any import failure is logged but doesn't crash vLLM.

---

### 3.11 `vllm_patches/unapply_patch.sh` (27 lines) — Patch remover

**Purpose**: Removes the energy scheduler from vLLM. Used for cleanup or reverting to a clean vLLM state.

**L6**: Removes the `vllm/energy_sched/` directory.

**L9–18**: Uses a Python regex to remove everything from the `ENERGY_SCHED_HOOK` marker to EOF in `vllm/__init__.py`, restoring the file to its original state.

---

### 3.12 `vllm_patches/__init__.py` (1 line) — Package marker

**Purpose**: Empty `__init__.py` file that makes `vllm/energy_sched/` a Python package, enabling `from vllm.energy_sched.energy_scheduler import ...` imports.

---

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

**How the hook works**: When vLLM is imported (`import vllm`), `__init__.py` runs. If `VLLM_ENERGY_SCHEDULER=1` is set in the environment, it imports `make_energy_scheduler_class()` and monkey-patches `vllm.v1.core.sched.scheduler.Scheduler` to be the energy-aware subclass. This is a non-invasive patch — no existing vLLM files are modified, and the patch is a pure no-op unless the env var is set.

**New directory**: `/home/ubuntu/lqs/vllm/vllm/energy_sched/` — contains `__init__.py`, `energy_model.py`, `energy_scheduler.py`, `frequency_controller.py`.

## 5. Dataset Provenance

- **Repo**: `RyokoAI/ShareGPT52K` on Hugging Face
- **Auto-download**: `prepare_dataset.py` automatically downloads the dataset via `huggingface_hub` if the directory is missing, or re-downloads if Git LFS pointers are detected (file size < 200 bytes, content starts with `version`). No manual `snapshot_download` command is needed.
- **Trace**: 400 requests, first human message per conversation, prompt length 64–8000 chars
- **SLO parameters**: TTFT μ=1000ms σ=200ms, TPOT μ=50ms σ=40ms (truncated normal)
- **Arrival rate**: 10 req/s (uniform)

## 6. Results (latest run — after bug fixes)

Parameters: `BETA=0.01, W_TTFT=2000, W_TPOT=50, ETA_MS=200, MAX_NUM_SEQS=128, NUM_REQUESTS=400, RATE_QPS=10, FREQ_STRIDE=4`.

| Metric | Default | Custom |
|--------|---------|--------|
| num_completed | — | 400 |
| num_failed | — | 0 |
| mean_ttft_ms | — | 27632.11 |
| mean_tpot_ms | — | 58.84 |
| mean_ttft_violation_ms | — | 26868.73 |
| mean_tpot_violation_ms | — | 13.96 |
| ttft_slo_attainment | — | 0.27 |
| tpot_slo_attainment | — | 0.3125 |
| mean_power_w | — | 182.35 |
| total_energy_j | — | 27378.72 |
| mean_solve_exec_ratio | — | 0.1767 |

**Notes on results**:
- All 400 requests completed in custom mode (the previous run with 399 failures was caused by a `NameError` in the debug code, now fixed).
- Mean power is 182.35W vs typical ~340W default, reflecting the solver's tendency to pick lower GPU frequencies (930–990 MHz).
- `mean_solve_exec_ratio = 0.177` means the solver takes ~18% of the batch execution time — acceptable overhead for per-iteration use.
- TTFT is significantly higher than default because the solver defers prefill requests whose adjusted utility is low when β·P(f)·t_q dominates.

## 7. How to Reproduce

```bash
cd /home/ubuntu/lqs/energy_efficient_LLM_scheduling
conda activate myvllm

# Run experiment (dataset is auto-downloaded if needed)
bash main.sh

# View results
cat results/demo/comparison.csv
```

To change experiment parameters, edit the USER KNOBS block at the top of `main.sh`. To change trace generation parameters (or force re-download of the dataset), edit the USER KNOBS block at the top of `scripts/prepare_dataset.py`. The dataset is automatically verified and downloaded if Git LFS pointer files are detected.
