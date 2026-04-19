# Experiment Instruction — Energy-Efficient LLM Scheduling on vLLM

> **This file is the ONLY input the server-side Claude Code agent
> receives.** No PDF, no profiling folders, no helper scripts, no
> `main.sh`, no `fitted_params.json` are uploaded. Everything the
> scheduler needs — the formulation, the fitted coefficients, the
> algorithm, the file layout — is contained **inside this document**.
> The agent must *create* every script, patch, and helper listed in §7
> from scratch, following the specifications here.

The server-side Claude Code agent that runs on the host with vLLM and
the A800 GPU should **follow this file end to end** to:

1. Read the formulation in §2–§3 and the fitted coefficients in §4.
   All numeric values are transcribed **literally** in this document —
   do not look for external JSON files.
2. Create every script / patch / helper listed in §7 inside the working
   directory (henceforth "the project root" — typically the folder where
   this file was uploaded). The agent creates these files; the user
   does not supply them.
3. Download the ShareGPT52K dataset from Hugging Face into
   `data/sharegpt52k/` under the project root (see §6).
4. Patch the vLLM source tree at `/home/ubuntu/lqs/vllm` in a
   **non-invasive** way: the patch must be a pure no-op unless the env
   variable `VLLM_ENERGY_SCHEDULER=1` is set.
5. Run the experiment with the `main.sh` master script **that the agent
   itself just created in step 2**.
6. At the end, write a companion `experiment.md` file that documents
   the **actual** workflow, the commands executed, and every file the
   agent created or modified (including the line ranges inside vLLM
   that were changed).

Do **all** work inside the conda environment `myvllm` (it already has
`huggingface_hub`, `datasets`, `aiohttp`, `requests`, `pynvml`, and the
editable vLLM install). If any of those are missing, install them with
`pip install <pkg>` inside that env.

Starting state assumption: the project root contains **only this
`experiment_instruction.md`** (plus whatever hidden files Claude Code
creates for itself). If earlier runs left behind an `experiment/`
folder or `experiment.md`, remove them with
`rm -rf experiment experiment.md` before you start — but this is a
courtesy, not a requirement.

---

## 1. Background and goal

We want to run a single vLLM server that serves Qwen3-14B on A800 and
compare two schedulers on exactly the same workload:

* **Baseline**: vLLM's default scheduler, GPU clocks not locked.
* **Ours**: an energy-aware scheduler that solves the per-iteration
  optimisation in §2 below using the *frequency-first* solution in §3,
  and locks the SM clock via `pynvml` on every iteration.

The reported metrics (per run) are:

* mean TTFT (ms), mean TPOT (ms)
* mean TTFT violation: `mean_n [max(0, TTFT_actual_n − TTFT_req_n)]`
* mean TPOT violation: `mean_n [max(0, TPOT_actual_n − TPOT_req_n)]`
* mean GPU power (W), total energy (J)
* TTFT / TPOT SLO attainment (fraction)
* **mean solve-to-exec ratio**:
  `mean_i [solve_ms_i / exec_ms_i]` averaged across every scheduling
  iteration `i`, where `solve_ms_i` is the wall-clock cost of the
  frequency-first `(B, f)` solve on iteration `i` and `exec_ms_i` is
  the wall-clock cost of the batch that solve produced (i.e. the gap
  between `schedule()` exit and the next `schedule()` entry). This
  metric is only meaningful for the custom scheduler; **report 0 for
  vLLM's default scheduler**.

---

## 2. Formulation (Section 1 of the paper, transcribed verbatim)

All the notation below is self-contained; you do NOT need any PDF.

### 2.1 Per-iteration optimisation

At iteration `i` the scheduler sees two queues:

* `R_i` — running queue (decode requests that already produced ≥1 output token).
* `W_i` — waiting queue (prefill requests; no output token yet).

For each request `n ∈ R_i ∪ W_i` let

* `T_{i,n}` — total waiting time accumulated since `n` was last executed,
  in **milliseconds**.
* `TTFT_n`, `TPOT_n` — per-request deadlines (ms) supplied by the client.
* `w_n` — per-request priority (default 1.0).

Deadline for this iteration:

```
deadline_n = TTFT_n,  if the next output token index is 1 (i.e., n ∈ W_i)
           = TPOT_n,  otherwise                       (n ∈ R_i)
```

Baseline utility:

```
r_n = w_n · w_TTFT     if n is prefill
    = w_n · w_TPOT     if n is decode
```

Instantaneous SLO-adherence utility:

```
f_{i,n} = r_n · exp(-(deadline_n − T_{i,n}))        (eq. 1)
```

**Units convention used by our implementation**: feed `(deadline_n −
T_{i,n})` to `exp()` in **seconds** (i.e., divide the ms-difference by
1000). This keeps the exponent in a numerically sane range for typical
SLOs (TTFT ≈ 3 s, TPOT ≈ 200 ms).

Per-iteration problem (same as the paper, with `ET_i(B, f)` now given
by (**) in §2.2):

```
max_{B, f}   sum_{n ∈ B} f_{i,n}  −  β · P(f) · ET_i(B, f)           (eq. 4)
s.t.         sum_{n ∈ B} ℓ_{i,n}  ≤  L_max                            (eq. 5)
             ET_i(B, f)            ≤  η                               (eq. 6)
```

where `ℓ_{i,n} = l_q` is the number of tokens processed in this
iteration, `β` is the energy balance parameter, and `η` is a
per-iteration time budget. `η` is a **tunable parameter** and its
default is set very large (effectively `+∞`) so that eq. 6 is inactive
— energy/utility trade-offs are driven by the objective alone. Ablate
by lowering `η` to a finite millisecond value via the `ETA_MS` knob
(§7.1) if you want to cap worst-case iteration time.

Because `ET_i(B, f)` contains the two indicator terms from §2.2, both
the objective and the time constraint pick up a **step change** when
the first prefill (or first decode) is added to `B`. §3 shows how the
frequency-first algorithm still decomposes cleanly by enumerating the
batch-type mask `M ∈ {prefill_only, decode_only, mixed}` in an outer
loop.

### 2.2 Batch execution time model (Route B+, 9-parameter)

The profiling stage re-fitted the batch-latency model to **Route B+**,
which splits the single lumped overhead `t_c` of the old 7-parameter
fit into (i) a per-batch prefill-side overhead `w_pf`, (ii) a per-batch
decode-side overhead `w_dec`, and (iii) a small residual constant `t_c`.
Per-type MAPE drops from ~14 % to ~5 %, so **the scheduler must use this
form**.

Let `l_q = l_{i,n}` be the tokens processed this iteration for request
`n` (= input length for prefill, = 1 for decode), and
`l_kv = l_{i,n}^{kv}` the KV-cache length at the start of the iteration.

Define the **per-request** terms (these add up over the batch):

```
t_q(n, f) = ( a_p · l_q^2  +  b_p · l_q · l_kv  +  c_p · l_q ) / f      (prefill)
          = ( a_d · l_kv   +  b_d )                   / f^α             (decode,  l_q = 1)
```

and the **per-batch** (type-indicator) overhead:

```
T_ovh(B, f) = w_pf · 1{∃ n ∈ B : n is prefill} / f
            + w_dec · 1{∃ n ∈ B : n is decode}  / f^α
```

The batch execution time used throughout is

```
ET_i(B, f) = sum_{n ∈ B} t_q(n, f)  +  T_ovh(B, f)  +  t_c           (**)
```

> The two indicators fire once per batch — not once per request.
> Physically, `w_pf` and `w_dec` capture the per-iteration weight-read /
> kernel-launch cost of traversing the 40 transformer layers once in
> prefill mode (or once in decode mode); they do not scale with `|B|`.
> Consequently, **splitting the batch into pure-prefill and pure-decode
> iterations incurs both overheads**, while a mixed batch pays each
> overhead at most once.

For the knapsack view in eq. 4 it is convenient to absorb the
per-batch overhead into the objective and the time constraint
explicitly — §3 does this by first enumerating the batch-type mask
`M`, which turns `T_ovh` into a known constant per `(f, M)` pair.

Symbol: throughout we write `t_q(n, f)` for the per-request piece and
`T_ovh(B, f)` for the batch-level piece.

### 2.3 Power model

```
P(f) = k3 · f^3  +  k2 · f^2  +  k1 · f  +  k0                 (Watts, f in MHz)
```

---

## 3. Frequency-first solution (Section 1.1 of the paper, adapted to Route B+)

Direct enumeration over `(B, f)` is `O(|F|_sub · 2^{|N_i|})`, which
grows sharply past `|N_i| ≈ 20`. We instead use the Frequency-first
decomposition. The only complication introduced by Route B+ is that
the per-batch indicators `w_pf` / `w_dec` (§2.2) make the adjusted
utility of a single request depend on *whether the batch already
contains any prefill / decode*. We sidestep this with an extra outer
enumeration over the three relevant batch-type masks

```
M ∈ { prefill_only, decode_only, mixed }
```

Given `M`, the per-batch overhead becomes a known constant
`T_ovh(M, f)` and the problem collapses back to a standard 2-D
knapsack over the candidate subset.

### 3.1 Algorithm

For every candidate SM frequency `f ∈ F[::FREQ_STRIDE]` and every mask
`M`:

1. **Define per-request adjusted utility.**
   ```
   v_{i,n}(f) = f_{i,n}  −  β · P(f) · t_q(n, f)
   ```
   `t_q(n, f)` is the **per-request** piece only — the per-batch
   overhead is accounted for separately below.

2. **Restrict the candidate set** to
   ```
   C(M) = { all prefill requests in W_i }                if M = prefill_only
        = { all decode  requests in R_i }                if M = decode_only
        = R_i ∪ W_i                                      if M = mixed
   ```

3. **Compute the batch overhead and tightened time budget.**
   ```
   T_ovh(M, f) = (w_pf / f)    · 1{M contains a prefill}
               + (w_dec / f^α) · 1{M contains a decode}
   η'(M, f)    = η − T_ovh(M, f) − t_c        # budget left for Σ t_q
   ```
   If `η'(M, f) ≤ 0` this `(f, M)` pair is infeasible — skip it.

4. **Solve the 2-D 0/1 knapsack restricted to `C(M)`:**
   ```
   B(f, M) = argmax_{B ⊆ C(M)}  sum_{n ∈ B} v_{i,n}(f)
            s.t.                sum_{n ∈ B} ℓ_{i,n}     ≤ L_max,
                                sum_{n ∈ B} t_q(n, f)  ≤ η'(M, f)
   ```

5. **Evaluate the full objective.**
   ```
   J(f, M) = sum_{n ∈ B(f, M)} v_{i,n}(f)
             − β · P(f) · ( T_ovh(M, f) + t_c )
   ```

Then pick `(f*, M*) = argmax_{f, M} J(f, M)` and `B* = B(f*, M*)`.

> If `B(f, M)` comes back empty or violates the mask-consistency check
> (e.g. `mixed` with only decodes chosen), treat that `(f, M)` as
> infeasible and skip. The `M = decode_only` case always dominates a
> would-be empty-prefill `mixed` case, so no opportunity is lost.
> The empty batch `B = ∅` with `J = 0` is always feasible and serves
> as the safety fallback when nothing else beats it.

**Knapsack solver** — use the utility-density greedy (pseudo-optimal
for this formulation and `O(|C(M)| · log |C(M)|)` per `(f, M)` pair):
sort by `v_{i,n}(f) / max(ℓ_{i,n} / L_max, t_q(n, f) / η'(M, f))` and
add while both constraints hold. Skip any request with `v_{i,n}(f) ≤ 0`.

### 3.2 Complexity

Outer enumeration cost: `|F|_sub · 3`. With the default
`FREQ_STRIDE=4` (⌈82/4⌉ = 21 candidate frequencies) this is 63
knapsack solves per scheduling iteration, well within the
millisecond-scale budget the paper's algorithm targets. Each knapsack
is `O(|C(M)| · log |C(M)|)`, so the total per-iteration cost is
roughly `|F|_sub · |N_i| · log|N_i|` — linear in `|N_i|`, versus the
`2^{|N_i|}` of a direct search.

**Timing instrumentation.** The scheduler measures the wall-clock
cost of every `solve()` call (`solve_ms`) and the wall-clock cost of
the batch the call produced (`exec_ms`, the elapsed time between
successive `schedule()` entries — see §7.9). Both are written to the
per-iteration log so that `metrics_collector.py` can report the mean
`solve_ms / exec_ms` ratio (see §1).

### 3.3 KV-cache post-filter (paper does not constrain this; we add it)

After `B*` is chosen, query vLLM's KV-cache manager for the number of
free blocks. If the blocks required by `B*` exceed the free pool,
repeatedly remove from `B*` the request with the smallest
`v_{i,n}(f*)` until `B*` fits. Running-queue requests not selected in
`B*` this iteration follow vLLM's existing **swap-to-CPU** preemption
path.

### 3.4 Frequency switching

Use `pynvml.nvmlDeviceSetGpuLockedClocks(handle, f*, f*)` right before
the iteration executes. Cache the last value and skip the NVML call if
`f*` is unchanged. Overhead is tens of microseconds per call — cheap
enough to do every iteration.

---

## 4. Fitted coefficients (transcribed from our profiling pipeline)

The numbers below were fitted off-line on the same A800-SXM4-80GB GPU
that will run the experiment, using the conditions described in §5
(enforce-eager, no chunked prefill, no prefix cache). **Use them as-is
— you do not need any external JSON file.** As an optional convenience,
the scheduler will prefer updated values if `VLLM_ENERGY_LATENCY_JSON`
or `VLLM_ENERGY_POWER_JSON` env variables point at a JSON file written
in the same schema, but this is never required for the default run.

### 4.1 Latency model — Route B+ (9-parameter)

These are the 9-parameter Route B+ coefficients; they plug directly into
the batch-time equation (**) of §2.2.

| Symbol | Value                   | Role                                         |
|--------|-------------------------|----------------------------------------------|
| `a_p`  |  4.791065541701025e-03  | prefill `l_q²` (attention-score cost)        |
| `b_p`  |  0.0                    | prefill cross-term `l_q·l_kv` (prefix cache off ⇒ unidentified) |
| `c_p`  |  1.3650620923913607e+02 | prefill `l_q` (per-token prefill compute)    |
| `w_pf` |  1.4999985410892014e+04 | per-batch prefill overhead (weight-load / launch) |
| `w_dec`|  1.4999998752209698e+04 | per-batch decode  overhead (weight-load / launch) |
| `a_d`  |  1.9294172833774345e-01 | decode `l_kv` (per-request attention-read)   |
| `b_d`  |  5.0502340019606976e+01 | decode constant (per-request)                |
| `α`    |  0.9735669988793928     | decode frequency exponent (`1/f^α`)          |
| `t_c`  |  4.652569884043852      | residual batch-level constant                |

Units: `f` in MHz, result in **milliseconds**. `b_p` is zero because
prefix caching was disabled during profiling, so `l_kv = 0` for every
prefill request in the training data and the cross-term is not
identifiable. On the full 101 k profiling set Route B+ achieves
overall MAPE ≈ 4.82 % (prefill 5.01 %, mixed 5.65 %, decode 4.79 %),
versus the 13.73 % MAPE of the earlier 7-parameter lumped fit — in
particular the decode Spearman ρ with measured wall-time rose from
0.58 → 0.97, which is the rank-ordering quality the scheduler cares
about.

`w_pf` and `w_dec` both sit at the fitting-heuristic ceiling of 1.5 × 10⁴;
physically this is the ~15 ms cost of reading the full 40-layer weight
set once per batch at 1 GHz, and it is the very cost the old 7-parameter
fit was forced to absorb into `t_c` (hence that model's 21 ms lumped
constant). Because the term fires **once per batch, not once per
request**, §3 enumerates the three relevant batch-type masks `M`.

### 4.2 Power model (cubic fit on measured wall-clock power vs SM clock)

| Symbol | Value              |
|--------|--------------------|
| `k3`   |  1.711824e-07      |
| `k2`   | -3.252635e-04      |
| `k1`   |  3.194042e-01      |
| `k0`   |  3.789920e+01      |

Units: `f` in MHz, result in **Watts**. R² = 0.9904, MAPE = 2.92 %.

### 4.3 GPU frequency candidates

A800-SXM4-80GB supported SM clocks (MHz), available via NVML:
`210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405,
420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 585, 600, 615,
630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825,
840, 855, 870, 885, 900, 915, 930, 945, 960, 975, 990, 1005, 1020,
1035, 1050, 1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1185, 1200,
1215, 1230, 1245, 1260, 1275, 1290, 1305, 1320, 1335, 1350, 1365, 1380,
1395, 1410`.

The profiling fit is trustworthy on `[510, 1335] MHz`; the scheduler
can still choose lower clocks but predictions for the latency model
outside that range are extrapolations.

Memory clock is hardware-fixed at 1593 MHz on A800.

---

## 5. Server-side environment (must match the profiling conditions)

| Item              | Value                                       |
|-------------------|---------------------------------------------|
| Host              | Linux with NVIDIA A800-SXM4-80GB            |
| Conda env         | `myvllm`                                    |
| vLLM source       | `/home/ubuntu/lqs/vllm` (editable install)  |
| Model             | `/home/ubuntu/lqs/LLM_model` (Qwen3-14B)    |
| vLLM flags (always) | `--enforce-eager` (disables CUDA graphs),<br>`--no-enable-chunked-prefill`,<br>`--no-enable-prefix-caching` |
| Root access       | required for `nvmlDeviceSetGpuLockedClocks` on most hosts (or grant `CAP_SYS_NICE`) |

Everything — launching vLLM, sending the workload, logging power,
collecting metrics — runs inside `conda activate myvllm`. **Do not
create a new env**: the user has already installed everything needed
there (huggingface_hub, datasets, aiohttp, requests, pynvml, vllm).

---

## 6. Dataset

Download RyokoAI/ShareGPT52K into this folder once:

```bash
# Inside the repo root (the folder containing experiment_instruction.md)
mkdir -p data/sharegpt52k
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="RyokoAI/ShareGPT52K",
    repo_type="dataset",
    local_dir="data/sharegpt52k",
    local_dir_use_symlinks=False,
)
print("Downloaded to:", path)
PY
```

The dataset ships as JSON files; the preprocessing script (§7.2) picks
the **first human message** of each conversation as the prompt.

---

## 7. What you (server-side Claude Code) must create

Starting from a project root that contains only this
`experiment_instruction.md`, create the files below in these exact
relative paths. Use the function signatures and algorithms described;
feel free to add logging, docstrings, and type hints.

```
├── main.sh                          # master runner -- user-editable knobs at the top
├── data/sharegpt52k/                # downloaded dataset (step 6)
├── trace.jsonl                      # synthesised per-request workload (step 7.2)
├── scripts/
│   ├── prepare_dataset.py           # 7.2
│   ├── workload_sender.py           # 7.3
│   ├── power_logger.py              # 7.4
│   ├── metrics_collector.py         # 7.5
│   └── compare_results.py           # 7.6
├── vllm_patches/
│   ├── energy_model.py              # 7.7 — fitted coefficients + per-request time
│   ├── frequency_controller.py      # 7.8 — pynvml wrapper
│   ├── energy_scheduler.py          # 7.9 — the frequency-first scheduler
│   ├── __init__.py
│   ├── apply_patch.sh               # 7.10 — installs the package into vLLM
│   └── unapply_patch.sh
└── results/<tag>/                   # created per run by main.sh
```

### 7.1 `main.sh` — master runner (REQUIRED NAME)

A Bash script at the project root with:

* `set -euo pipefail`.
* A **clearly-labeled "USER KNOBS" block** near the top with default
  values for every parameter listed below (so the user can edit and
  re-run). All knobs are also accepted as `--key value` CLI overrides.
* Sources conda (`~/miniconda3/etc/profile.d/conda.sh` or
  `/home/ubuntu/miniconda3/etc/profile.d/conda.sh`, first one that
  exists) and `conda activate myvllm`.
* Calls `bash vllm_patches/apply_patch.sh /home/ubuntu/lqs/vllm`.
* Calls `scripts/prepare_dataset.py` to produce `trace.jsonl` (skipped
  if the file already exists and `--force-trace` is not set).
* Runs **two sub-experiments** (or one, depending on `--mode`):
  default scheduler then custom scheduler, on the same trace.
  For each:
    1. Launches the vLLM server with the three fixed flags in §5. In
       `custom` mode, exports the `VLLM_ENERGY_*` env vars listed below
       — including `VLLM_ENERGY_ITER_LOG=results/$TAG/iter_custom.log`
       — **before** the server starts.
    2. Starts the power logger in the background.
    3. Runs the workload sender.
    4. Stops the power logger and the server; resets GPU clocks.
    5. Runs `metrics_collector.py` to produce `summary_<label>.json`.
       In `custom` mode it passes `--iter-log results/$TAG/iter_custom.log`;
       in `default` mode it passes no `--iter-log` (the ratio is
       reported as 0).
* Finally calls `compare_results.py` to produce `comparison.csv`.

**Required user knobs** (all must appear and be tunable):

| Knob                 | Env / var name              | Default |
|----------------------|-----------------------------|---------|
| Results tag          | `TAG`                       | `default_run` |
| Mode                 | `MODE`                      | `both` (one of `default` / `custom` / `both`) |
| GPU index            | `GPU_INDEX`                 | `0` |
| Port                 | `PORT`                      | `8000` |
| vLLM source dir      | `VLLM_DIR`                  | `/home/ubuntu/lqs/vllm` |
| Model dir            | `MODEL_DIR`                 | `/home/ubuntu/lqs/LLM_model` |
| Served model name    | `MODEL_NAME`                | `default` |
| # requests           | `NUM_REQUESTS`              | `1000` |
| Arrival rate (req/s) | `RATE_QPS`                  | `4` |
| TTFT μ (ms)          | `TTFT_MEAN_MS`              | `3000` |
| TTFT σ (ms)          | `TTFT_STD_MS`               | `500` |
| TPOT μ (ms)          | `TPOT_MEAN_MS`              | `200` |
| TPOT σ (ms)          | `TPOT_STD_MS`               | `40` |
| output tokens min    | `MIN_OUT_TOK`               | `64` |
| output tokens max    | `MAX_OUT_TOK`               | `512` |
| Seed                 | `TRACE_SEED`                | `42` |
| β                    | `BETA`                      | `1.0` |
| `w_TTFT`             | `W_TTFT`                    | `1.0` |
| `w_TPOT`             | `W_TPOT`                    | `1.0` |
| η (ms; effectively ∞ ⇒ eq. 6 inactive) | `ETA_MS`  | `1e9` |
| `L_max`              | `LMAX`                      | `0` (⇒ inherit vLLM `max_num_batched_tokens`) |
| freq-candidate stride| `FREQ_STRIDE`               | `4` |
| `max_model_len`      | `MAX_MODEL_LEN`             | `8192` |
| `max_num_seqs`       | `MAX_NUM_SEQS`              | `64` |
| gpu-mem util         | `GPU_MEM_UTIL`              | `0.90` |
| power sample period  | `POWER_INTERVAL_S`          | `0.1` |

The "custom" mode exports these env vars before launching the server so
the scheduler picks them up:

```
VLLM_ENERGY_SCHEDULER=1
VLLM_ENERGY_BETA=$BETA
VLLM_ENERGY_W_TTFT=$W_TTFT
VLLM_ENERGY_W_TPOT=$W_TPOT
VLLM_ENERGY_LMAX=$LMAX
VLLM_ENERGY_FREQ_STRIDE=$FREQ_STRIDE
VLLM_ENERGY_ETA_MS=$ETA_MS          # default 1e9 ms ⇒ η effectively ∞
VLLM_ENERGY_GPU_INDEX=0     # always 0 inside CUDA_VISIBLE_DEVICES
VLLM_ENERGY_ITER_LOG=results/$TAG/iter_custom.log
```

The `default` mode does **not** set `VLLM_ENERGY_ITER_LOG` (the
default scheduler never touches it). `metrics_collector.py` detects
the file's absence and reports `mean_solve_exec_ratio = 0.0`.

Provide `bash main.sh --help` that prints the knob table.

### 7.2 `scripts/prepare_dataset.py`

Reads `data/sharegpt52k/*.json`, keeps conversations whose first human
message has length in `[min_prompt_chars, max_prompt_chars]`, shuffles
with the given seed, and writes `trace.jsonl` with one record per line:

```
{"id": "req_000001",
 "arrival_s": 0.25,          # i/RATE_QPS for record i (uniform arrivals)
 "prompt": "...",            # raw text passed to vLLM
 "max_tokens": 312,          # uniform int in [MIN_OUT_TOK, MAX_OUT_TOK]
 "ttft_ms": 3214.5,          # truncated normal, mean=TTFT_MEAN_MS, std=TTFT_STD_MS, > 1
 "tpot_ms": 187.3,           # truncated normal, mean=TPOT_MEAN_MS, std=TPOT_STD_MS, > 1
 "w_n": 1.0}                 # default priority
```

CLI flags must include: `--output`, `--num-requests`, `--rate-qps`,
`--ttft-mean-ms`, `--ttft-std-ms`, `--tpot-mean-ms`, `--tpot-std-ms`,
`--min-output-tokens`, `--max-output-tokens`, `--min-prompt-chars`
(default 64), `--max-prompt-chars` (default 8000), `--seed`,
`--dataset-dir` (default `data/sharegpt52k`).

### 7.3 `scripts/workload_sender.py`

Async replay against an OpenAI-compatible vLLM endpoint
(`/v1/completions`, `stream=true`). For each record in `trace.jsonl`:

* Wait until wall-clock `t_rel >= arrival_s`.
* POST with `max_tokens=record.max_tokens`, `temperature=0`,
  `stream=true`, and
  ```json
  "extra_body": {"extra_args":
     {"ttft_ms": ..., "tpot_ms": ..., "w_n": ...}}
  ```
  — vLLM forwards `extra_args` into `SamplingParams.extra_args` which
  the energy scheduler reads per-request.
* Record per-chunk timestamps, compute per-request:
  * `ttft_ms` = first-chunk-time − send-time (ms).
  * `tpot_ms` = mean of successive inter-chunk gaps (ms).
  * `num_output_tokens` = number of non-empty text chunks received.
* Write each completed record as one JSONL line into `results.jsonl`
  with the full metadata (send time, complete time, SLOs, status, etc.)
  so `metrics_collector.py` can process it.

CLI flags: `--trace`, `--endpoint`, `--model`, `--output`,
`--max-concurrency` (default 256).

### 7.4 `scripts/power_logger.py`

Uses `pynvml` to sample every `--interval` seconds (default 0.1) and
write a CSV with columns
`timestamp_s, power_w, sm_clock_mhz, utilization_pct`.

Must flush after every row and stop cleanly on SIGTERM / SIGINT.

### 7.5 `scripts/metrics_collector.py`

Input: `--requests results.jsonl --power power.csv
--iter-log iter.log --label <default|custom> --output summary.json`.

`--iter-log` may be missing (`default` mode never produces one) — the
collector must treat that case as "no iteration data" and emit the
ratio as **0.0** (per §1).

Output JSON fields:

```
label, num_completed, num_failed,
mean_ttft_ms,           mean_tpot_ms,
mean_ttft_violation_ms, mean_tpot_violation_ms,
ttft_slo_attainment,    tpot_slo_attainment,
mean_power_w,           total_energy_j,
mean_solve_exec_ratio        # mean_i [solve_ms_i / exec_ms_i]; 0 for default
```

Violations are `mean_n [max(0, actual − SLO)]` over *completed* requests
(status 200, no error, and TTFT/TPOT not None). Power is integrated by
trapezoidal rule over the window `[first-send, last-complete]`, using
the power CSV's wall-clock timestamps.

The `mean_solve_exec_ratio` field is computed as:

```python
def solve_exec_ratio(iter_log_path: Optional[str]) -> float:
    if not iter_log_path or not os.path.exists(iter_log_path):
        return 0.0                             # default scheduler → 0
    ratios = []
    with open(iter_log_path) as f:
        for line in f:
            rec = json.loads(line)
            solve = rec.get("solve_ms")
            exec_ = rec.get("exec_ms")
            if solve is None or exec_ is None or exec_ <= 0.0:
                continue                       # first/last iter, or no batch
            ratios.append(solve / exec_)
    return float(sum(ratios) / len(ratios)) if ratios else 0.0
```

### 7.6 `scripts/compare_results.py`

Reads two summary JSONs and prints a side-by-side table; also writes
`comparison.csv` with header `metric,default,custom` and one row per
metric in §7.5's listed order. The row for `mean_solve_exec_ratio`
must appear in the output even though it is `0.0` for `default` —
this is the explicit contract from §1.

### 7.7 `vllm_patches/energy_model.py`

Define two dataclasses and helpers. The latency dataclass carries the
full **9-parameter Route B+** coefficient set and splits the helpers
into a per-request piece (`t_q`) and a per-batch piece
(`T_ovh`), matching the decomposition in §2.2 (**).

```python
@dataclass
class LatencyParams:
    a_p:   float = 4.791065541701025e-03
    b_p:   float = 0.0
    c_p:   float = 1.3650620923913607e+02
    w_pf:  float = 1.4999985410892014e+04    # per-batch prefill overhead
    w_dec: float = 1.4999998752209698e+04    # per-batch decode  overhead
    a_d:   float = 1.9294172833774345e-01
    b_d:   float = 5.0502340019606976e+01
    alpha: float = 0.9735669988793928
    t_c:   float = 4.652569884043852

    @classmethod
    def from_json(cls, path): ...           # OPTIONAL override — reads a
                                            # Route-B+ JSON file of the same
                                            # schema; unused by default

@dataclass
class PowerParams:
    k3: float = 1.711824e-07
    k2: float = -3.252635e-04
    k1: float = 3.194042e-01
    k0: float = 3.789920e+01
    def power_watts(self, f_mhz: float) -> float: ...
```

Also expose the following helpers. **Note**: `per_request_time_ms`
returns **only** the per-request `t_q(n, f)` piece — the per-batch
overhead is computed separately by `batch_overhead_ms` and added inside
`batch_time_ms`. The scheduler's adjusted-utility calculation (§7.9)
uses `per_request_time_ms`; the batch-level energy cost in the
objective uses `batch_overhead_ms + t_c`.

```python
def per_request_time_ms(latency: LatencyParams, freq_mhz: float,
                        is_prefill: bool, l_q: int, l_kv: int) -> float:
    """Per-request t_q(n, f) only. Excludes w_pf / w_dec / t_c."""
    if is_prefill:
        return (latency.a_p * l_q * l_q
                + latency.b_p * l_q * l_kv
                + latency.c_p * l_q) / freq_mhz
    # decode: l_q == 1 by construction
    return (latency.a_d * l_kv + latency.b_d) / (freq_mhz ** latency.alpha)

def batch_overhead_ms(latency: LatencyParams, freq_mhz: float,
                      has_prefill: bool, has_decode: bool) -> float:
    """T_ovh(B, f) = w_pf·1{has_prefill}/f + w_dec·1{has_decode}/f^alpha."""
    ovh = 0.0
    if has_prefill:
        ovh += latency.w_pf / freq_mhz
    if has_decode:
        ovh += latency.w_dec / (freq_mhz ** latency.alpha)
    return ovh

def batch_time_ms(latency: LatencyParams, freq_mhz: float,
                  requests: Sequence[Tuple[bool, int, int]]) -> float:
    """Total ET_i(B, f) = sum t_q + T_ovh(B, f) + t_c.
    Each request is (is_prefill, l_q, l_kv)."""
    if not requests:
        return 0.0
    has_prefill = any(r[0] for r in requests)
    has_decode  = any(not r[0] for r in requests)
    total = sum(per_request_time_ms(latency, freq_mhz, *r) for r in requests)
    total += batch_overhead_ms(latency, freq_mhz, has_prefill, has_decode)
    total += latency.t_c
    return total

def load_latency_params() -> LatencyParams   # default: dataclass defaults;
                                              # if $VLLM_ENERGY_LATENCY_JSON
                                              # is set and points at a file,
                                              # delegate to LatencyParams.from_json
def load_power_params()   -> PowerParams     # same convention for $VLLM_ENERGY_POWER_JSON
```

In production the JSON override is unused — the hard-coded defaults
**are** the Route B+ fit from §4.1 / §4.2.

`LatencyParams.from_json` must accept the Route B+ schema produced by
the profiling pipeline (top-level `"params"` dict with keys `a_p`,
`b_p`, `c_p`, `w_pf`, `w_dec`, `a_d`, `b_d`, `alpha`, `t_c`) and fall
back to the class defaults for any missing key (handy for loading
older 7-parameter dumps: they just inherit `w_pf = w_dec = 0`, making
the model degrade gracefully to the old lumped form).

### 7.8 `vllm_patches/frequency_controller.py`

`class FrequencyController` backed by `pynvml`, with:

* `__init__(gpu_index=0)` initialises NVML, queries supported graphics
  clocks with `nvmlDeviceGetSupportedGraphicsClocks`, falls back to
  `nvidia-smi --query-supported-clocks=gr` if needed.
* `supported_clocks() -> list[int]`.
* `set_frequency(f_mhz)` — snaps to the closest supported clock, calls
  `nvmlDeviceSetGpuLockedClocks(h, target, target)`, caches last value,
  returns True / False.
* `reset()` — `nvmlDeviceResetGpuLockedClocks`.

Also provide a module-level `get_controller()` singleton that reads
`VLLM_ENERGY_GPU_INDEX`.

### 7.9 `vllm_patches/energy_scheduler.py`

Two layers:

**(a) Pure-Python core (no vLLM imports)** — makes it unit-testable.

```python
@dataclass
class EnergySchedConfig:
    beta: float = 1.0
    w_ttft: float = 1.0
    w_tpot: float = 1.0
    eta_ms: float = 1e9             # very large ⇒ eq. 6 inactive by default
                                    # override via $VLLM_ENERGY_ETA_MS
    Lmax: int = 0                   # 0 → inherit at scheduler construction
    default_w_n: float = 1.0
    default_ttft_ms: float = 4000.0
    default_tpot_ms: float = 200.0
    freq_candidates: Optional[List[int]] = None
    freq_stride: int = 1
    log_every_n: int = 50
    iter_log_path: Optional[str] = None   # JSONL per-iteration log; see below
    @classmethod
    def from_env(cls): ...           # reads VLLM_ENERGY_* env vars

@dataclass
class ReqView:
    handle: Any                     # opaque vLLM Request
    is_prefill: bool
    l_q: int
    l_kv: int
    wait_ms: float                  # T_{i,n}
    deadline_ms: float              # TTFT or TPOT
    w_n: float
    kv_blocks_needed: int = 0

def instant_utility(r: ReqView, cfg) -> float:         # eq. 1, 2, 3
    r_n = r.w_n * (cfg.w_ttft if r.is_prefill else cfg.w_tpot)
    slack_ms = r.deadline_ms - r.wait_ms
    return r_n * math.exp(-slack_ms / 1000.0)          # slack in seconds

def adjusted_utility(r, cfg, f_mhz, latency, power) -> Tuple[float, float]:
    """Returns (v_{i,n}(f), t_q(n, f)).

    Per §3 the per-batch overhead w_pf / w_dec / t_c is accounted for
    *outside* this function (once per (f, M) pair) — this routine only
    combines the per-request t_q piece with the energy-cost term.
    """
    t_q = per_request_time_ms(latency, f_mhz, r.is_prefill, r.l_q, r.l_kv)
    f_in = instant_utility(r, cfg)
    v = f_in - cfg.beta * power.power_watts(f_mhz) * t_q
    return v, t_q

def greedy_knapsack_2d(reqs, values, times_ms, tokens,
                       Lmax, eta_ms) -> List[int]:
    """Utility-density greedy 2-D 0/1 knapsack.
    Returns indices of chosen items. Skip v_{i,n}(f) <= 0 items."""
    ...

class FrequencyFirstSolver:
    """Enumerates (f, M) ∈ F × {prefill_only, decode_only, mixed} and
    calls a 2-D 0/1 knapsack per pair. See §3."""
    def __init__(self, cfg, latency, power, freq_candidates): ...

    def solve(self, reqs, Lmax) -> Tuple[float, list, float]:
        """
        Returns (f_star, chosen_batch, et_pred).

        Pseudocode:

            best = (0.0, freq_candidates[-1], [], 0.0)  # empty-batch fallback
            has_prefill_any = any(r.is_prefill for r in reqs)
            has_decode_any  = any(not r.is_prefill for r in reqs)
            masks = []
            if has_prefill_any:                     masks.append("prefill_only")
            if has_decode_any:                      masks.append("decode_only")
            if has_prefill_any and has_decode_any:  masks.append("mixed")

            for f in freq_candidates[::cfg.freq_stride]:
                P_f = power.power_watts(f)
                # per-request values / times depend only on f, not M
                v_t = [adjusted_utility(r, cfg, f, latency, power) for r in reqs]
                for M in masks:
                    has_p = M in ("prefill_only", "mixed")
                    has_d = M in ("decode_only",  "mixed")
                    T_ovh = batch_overhead_ms(latency, f, has_p, has_d)
                    eta_left = cfg.eta_ms - T_ovh - latency.t_c
                    if eta_left <= 0:
                        continue
                    # restrict candidates to C(M)
                    idxs = [i for i, r in enumerate(reqs)
                            if (M != "prefill_only" or r.is_prefill)
                            and (M != "decode_only"  or not r.is_prefill)]
                    values  = [v_t[i][0] for i in idxs]
                    times   = [v_t[i][1] for i in idxs]
                    tokens  = [reqs[i].l_q   for i in idxs]
                    picked_local = greedy_knapsack_2d(
                        idxs, values, times, tokens, Lmax, eta_left)
                    picked = [idxs[j] for j in picked_local]
                    # mask-consistency check for "mixed":
                    # require at least one prefill AND one decode in picked
                    if M == "mixed":
                        chosen_reqs = [reqs[i] for i in picked]
                        if not (any(r.is_prefill for r in chosen_reqs)
                                and any(not r.is_prefill for r in chosen_reqs)):
                            continue
                    sum_v = sum(v_t[i][0] for i in picked)
                    J = sum_v - cfg.beta * P_f * (T_ovh + latency.t_c)
                    if J > best[0]:
                        et_pred = (sum(v_t[i][1] for i in picked)
                                   + T_ovh + latency.t_c)
                        best = (J, f, picked, et_pred)

            _, f_star, picked, et_pred = best
            return f_star, [reqs[i] for i in picked], et_pred
        """
        ...
```

**(b) vLLM integration — lazy-import factory** so the file imports
cleanly even when vLLM is missing:

```python
def make_energy_scheduler_class():
    from vllm.v1.core.sched.scheduler import Scheduler  # lazy
    class EnergyScheduler(Scheduler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cfg = EnergySchedConfig.from_env()
            self._latency = load_latency_params()
            self._power   = load_power_params()
            self._freq_ctl = get_controller()
            cands = self._cfg.freq_candidates or self._freq_ctl.supported_clocks() or [1410]
            self._solver = FrequencyFirstSolver(
                self._cfg, self._latency, self._power, cands)
            if self._cfg.Lmax <= 0:
                self._cfg.Lmax = int(getattr(
                    self.scheduler_config, "max_num_batched_tokens",
                    getattr(self.scheduler_config, "max_model_len", 8192)))
            # Iteration log: one JSON line per schedule() call.
            # Contains {iter, solve_ms, exec_ms, |B|, f_star, et_pred}.
            # Path comes from cfg.iter_log_path (default: $TAG_DIR/iter.log).
            self._iter_log = _open_iter_log(self._cfg.iter_log_path)
            self._prev_exit_t = None
            self._iter = 0

        def schedule(self):
            t_enter = time.monotonic()
            # exec_ms_i = wall-clock cost of the PREVIOUS batch, measured as
            # the gap between schedule() exit last iteration and schedule()
            # entry this iteration. This is the actual forward-pass cost
            # (model execution + update) observed by the engine loop.
            exec_ms = (
                (t_enter - self._prev_exit_t) * 1000.0
                if self._prev_exit_t is not None else None
            )

            now_ms = t_enter * 1000.0
            reqs   = self._build_request_views(now_ms)

            t_solve0 = time.monotonic()
            f_star, chosen, et_pred = self._solver.solve(reqs, self._cfg.Lmax)
            solve_ms = (time.monotonic() - t_solve0) * 1000.0

            self._freq_ctl.set_frequency(int(f_star))
            chosen = self._kv_evict(chosen, f_star)
            out = self._materialise_batch(chosen, f_star)

            # Log the ratio data *lagged* by one iteration: this iter's
            # solve_ms pairs with NEXT iter's exec_ms. We therefore flush
            # the PREVIOUS record (which was waiting for this exec_ms) now,
            # and stash the current solve_ms for next time.
            if self._prev_record is not None and exec_ms is not None:
                rec = self._prev_record
                rec["exec_ms"] = exec_ms
                self._iter_log.write(json.dumps(rec) + "\n")
                self._iter_log.flush()
            self._prev_record = {
                "iter": self._iter,
                "solve_ms": solve_ms,
                "batch_size": len(chosen),
                "f_star": int(f_star),
                "et_pred_ms": et_pred,
            }
            self._iter += 1
            self._prev_exit_t = time.monotonic()
            return out
        ...
    return EnergyScheduler
```

**Iteration log format** — one JSON object per line:
```
{"iter": 0, "solve_ms": 3.21, "exec_ms": 87.4, "batch_size": 6,
 "f_star": 1140, "et_pred_ms": 85.0}
```
The default path is `$RESULTS_DIR/iter_custom.log`; `main.sh` sets
`VLLM_ENERGY_ITER_LOG` to that path before launching the server.
The very first iteration has no `exec_ms` (nothing ran before it) and
is **not** written; the *last* iteration's `exec_ms` is likewise
unobservable through this mechanism and is dropped. Both losses are
`O(1)` across a run of thousands of iterations.

Helper methods inside `EnergyScheduler`:

* `_build_request_views(now_ms)` — iterate `self.waiting` (all prefill)
  and `self.running` (all decode) and build `ReqView` instances.
  Extract per-request SLOs from `req.sampling_params.extra_args`
  (fall back to cfg defaults if missing). Use
  `req.arrival_time` to compute `wait_ms`; read `num_prompt_tokens`
  for prefill `l_q` and `num_computed_tokens` for decode `l_kv`.
* `_kv_evict(chosen, f_mhz)` — if
  `sum(r.kv_blocks_needed) > kv_cache_manager.get_num_free_blocks()`,
  drop requests in ascending `v_{i,n}(f_mhz)` order until it fits.
* `_materialise_batch(chosen)` — temporarily restrict `self.waiting`
  and `self.running` to the chosen handles, call `super().schedule()`,
  then put the un-selected *waiting* requests back into `self.waiting`
  (so they are re-evaluated next iteration). Un-selected running
  requests are implicitly preempted (swap path) by the base class
  because they were missing from `self.running` during the super call.
  Finally restore any remaining running entries.

### 7.10 `vllm_patches/apply_patch.sh`

Bash script that:

1. Copies `__init__.py`, `energy_model.py`, `energy_scheduler.py`,
   `frequency_controller.py` into
   `$VLLM_DIR/vllm/energy_sched/` (new package, no existing vLLM file
   altered).
2. Appends the sentinel-guarded snippet below to
   `$VLLM_DIR/vllm/__init__.py` **once** (idempotent — check for the
   marker `# <<< ENERGY_SCHED_HOOK >>>` before appending):
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

`unapply_patch.sh` must remove both the package directory and the
appended block (use a Python regex to delete everything between the
marker and end-of-file / next sentinel).

> If your vLLM version locates `Scheduler` somewhere other than
> `vllm.v1.core.sched.scheduler`, search the tree for
> `class Scheduler` under `vllm/v1` and update the import path; the
> hook remains a 5-line conditional guarded by the env var so default
> behaviour is untouched.

---

## 8. Run order

The intended sequence, executed inside `conda activate myvllm`:

```bash
cd <project root>               # folder containing this file
rm -rf experiment experiment.md # clean up old draft (harmless if absent)

# 0. Author every file listed in §7 before running anything.
#    This is the agent's responsibility — the project root starts
#    with only experiment_instruction.md present.
#    Produce:  main.sh,
#              scripts/{prepare_dataset,workload_sender,power_logger,
#                       metrics_collector,compare_results}.py,
#              vllm_patches/{energy_model,frequency_controller,
#                            energy_scheduler,__init__}.py,
#              vllm_patches/{apply_patch,unapply_patch}.sh
#    Verify with:  ls main.sh scripts/*.py vllm_patches/*.{py,sh}

# 1. One-time setup (idempotent)
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='RyokoAI/ShareGPT52K', repo_type='dataset', \
                      local_dir='data/sharegpt52k', local_dir_use_symlinks=False)"
bash vllm_patches/apply_patch.sh /home/ubuntu/lqs/vllm

# 2. Default-vs-custom run
bash main.sh --tag demo \
    --num-requests 500 --rate-qps 4 \
    --ttft-mean-ms 3000 --tpot-mean-ms 200 \
    --beta 1.0 --w-ttft 1.0 --w-tpot 1.0 \
    --freq-stride 4 --mode both

# 3. Result
cat results/demo/comparison.csv
```

---

## 9. What to write at the end (`experiment.md`)

After the experiment finishes successfully, produce a
**`experiment.md`** in the project root that is a pure *retrospective*
record. It must contain:

1. A concise problem statement (copy §1 of this file, 1–2 paragraphs).
2. The exact commands you (server-side Claude) ran, top to bottom.
3. For every file under `scripts/`, `vllm_patches/` and `main.sh` you
   created:
   * its path, a one-line purpose, the number of lines, and a link or
     `cat` snippet of the key functions.
4. For every edit you made inside `/home/ubuntu/lqs/vllm`:
   * full file path;
   * exact line ranges of the inserted block (use `grep -n` to verify
     after installation);
   * the snippet itself (between the `# <<< ENERGY_SCHED_HOOK >>>`
     markers).
5. Dataset provenance: the Hugging Face repo id, commit hash of the
   snapshot (print `huggingface_hub.HfApi().dataset_info(...).sha`), and
   how many requests the trace contains.
6. Resulting numbers: copy the final `comparison.csv` contents into a
   Markdown table.
7. A short "How to reproduce" section ≤10 lines.

The goal is that someone who finds only `experiment.md` can redo the
exact run without reading `experiment_instruction.md`.

---

## 10. Sanity checklist (run these before declaring success)

* `python -c "import vllm; print(vllm.__version__)"` inside `myvllm`.
* `nvidia-smi --query-gpu=clocks.sm --format=csv,noheader` before and
  after a custom-mode run (should be reset to default at the end).
* `grep -n ENERGY_SCHED_HOOK /home/ubuntu/lqs/vllm/vllm/__init__.py`
  prints exactly one marker.
* With `VLLM_ENERGY_SCHEDULER` **unset**, launching vLLM normally
  (e.g. `python -m vllm.entrypoints.openai.api_server --model ... --enforce-eager`) behaves exactly as before — no scheduler change, no
  pynvml calls, no extra log lines.
* A dry run of `scripts/workload_sender.py` against a short
  (≤10-request) trace returns completions and writes `results.jsonl`
  rows with non-null `ttft_ms`.
* The final `comparison.csv` has two data columns (`default`, `custom`)
  and reasonable values (mean TTFT > 0, mean power ∈ [50, 350] W for
  A800, violations ≥ 0).
* The `mean_solve_exec_ratio` row in `comparison.csv` is exactly `0`
  under `default` and strictly positive (and well below 1.0 — otherwise
  the solver dominates the forward pass and `FREQ_STRIDE` should be
  raised) under `custom`.
* The iteration log `results/<TAG>/iter_custom.log` exists after a
  custom-mode run and has at least a few hundred lines; every line
  parses as JSON with both `solve_ms` and `exec_ms` present.

If any of these fail, fix the underlying issue **before** writing
`experiment.md`.
